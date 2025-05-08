import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List

import torch
from datasets import load_dataset
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          get_linear_schedule_with_warmup, set_seed)

# ─── Config ───────────────────────────────────────────────────────────────────

MODEL_NAME = "Qwen/Qwen3-0.6B-Base"
DATA_PATH = "./emoji_dataset"
OUTPUT_DIR = "./outputs"
VAL_RATIO = 0.1

# Sequence lengths
INPUT_MAX_LEN = 120
TARGET_MAX_LEN = 8
MAX_TOTAL_LEN = INPUT_MAX_LEN + TARGET_MAX_LEN

# Training hyperparams
BATCH_SIZE = 24
GRAD_ACCUM_STEPS = 1
LR = 1e-5
WARMUP_STEPS = 100
NUM_EPOCHS = 150

LOG_EVERY = 100             # steps
MAX_NEW_TOKENS = 5          # for eval‐generation
DEVICE = "cuda"             # single-GPU
USE_FP16 = False
SEED = 11944004

# ─── Prep ─────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)
set_seed(SEED)
torch.backends.cudnn.benchmark = True

# 1) Load & split raw dataset
raw = load_dataset(DATA_PATH, data_files={
                   "train": "train.jsonl", "validation": "val.jsonl"})
train_raw, eval_raw = raw["train"], raw["validation"]

# 2) Tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tokenizer.padding_side = "left"
tokenizer.truncation_side = "left"
if tokenizer.pad_token_id is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    print(f"Added pad token: {tokenizer.pad_token_id}")

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
model.resize_token_embeddings(len(tokenizer))

# ─── Tokenization ─────────────────────────────────────────────────────────────


def tokenize_example(ex: Dict) -> Dict:
    # encode input / output separately, no padding/truncation here
    in_ids = tokenizer.encode(
        ex["input"], add_special_tokens=False, truncation=True, max_length=INPUT_MAX_LEN)
    out_ids = tokenizer.encode(
        ex["output"], add_special_tokens=False, truncation=True, max_length=TARGET_MAX_LEN)
    return {
        "input_ids": in_ids,
        "output_ids": out_ids,
        "input_len": len(in_ids),
    }


train_tok = train_raw.map(
    tokenize_example,
    remove_columns=train_raw.column_names,
    batched=False,
)

# ─── Collate ─────────────────────────────────────────────────────────────────


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    # build input_ids + labels for each sample
    ids, masks, labels, in_lens = [], [], [], []
    for ex in batch:
        inp, out = ex["input_ids"], ex["output_ids"]
        combined = inp + out
        lab = [-100] * len(inp) + out

        # left-truncate to MAX_TOTAL_LEN
        if len(combined) > MAX_TOTAL_LEN:
            combined = combined[-MAX_TOTAL_LEN:]
            lab = lab[-MAX_TOTAL_LEN:]

        ids.append(combined)
        labels.append(lab)
        in_lens.append(min(ex["input_len"], MAX_TOTAL_LEN))

    # pad all sequences on the left to match longest in batch
    max_len = max(len(x) for x in ids)
    for i in range(len(ids)):
        pad_len = max_len - len(ids[i])
        ids[i] = [tokenizer.pad_token_id] * pad_len + ids[i]
        labels[i] = [-100] * pad_len + labels[i]

    # attention mask: 1 for non-pad
    masks = [[0]*pad_len + [1] * (len(ids[i]) - pad_len)
             for i, pad_len in enumerate([max_len - len(x) for x in ids])]

    batch_dict = {
        "input_ids":      torch.tensor(ids,    dtype=torch.long),
        "attention_mask": torch.tensor(masks,  dtype=torch.long),
        "labels":         torch.tensor(labels, dtype=torch.long),
        "input_lens":     torch.tensor(in_lens, dtype=torch.long),
    }
    return batch_dict


train_loader = DataLoader(
    train_tok,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    drop_last=True,
)

# ─── Optimizer & Scheduler ────────────────────────────────────────────────────

optimizer = AdamW(model.parameters(), lr=LR)
total_steps = (len(train_loader) // GRAD_ACCUM_STEPS) * NUM_EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=WARMUP_STEPS,
    num_training_steps=total_steps,
)

scaler = GradScaler()

# ─── Evaluation / Generation ─────────────────────────────────────────────────


def run_eval(gen_step: int):
    samples = eval_raw.select(range(min(len(eval_raw), BATCH_SIZE)))
    results = []
    for ex in samples:
        inp_text = ex["input"]
        labels_text = ex["output"]
        enc = tokenizer(inp_text, return_tensors="pt").to(DEVICE)

        with torch.no_grad(), autocast(device_type="cuda", dtype=torch.bfloat16 if not USE_FP16 else torch.float16):
            gen_ids = model.generate(
                **enc,
                max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=tokenizer.pad_token_id,
            )

        # slice off the prompt portion
        prompt_len = enc["input_ids"].shape[-1]
        new_ids = gen_ids[0, prompt_len:].tolist()
        new_text = tokenizer.decode(new_ids, skip_special_tokens=True)

        results.append({
            "input": inp_text,
            "new_tokens": new_text,
            "labels": labels_text,
        })

    out_path = os.path.join(OUTPUT_DIR, f"eval_step_{gen_step}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

# ─── Training Loop ────────────────────────────────────────────────────────────


global_step = 0
for epoch in range(1, NUM_EPOCHS + 1):
    for batch in train_loader:
        model.train()
        global_step += 1

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        with autocast(device_type="cuda", dtype=torch.bfloat16 if not USE_FP16 else torch.float16):
            loss = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            ).loss / GRAD_ACCUM_STEPS

        scaler.scale(loss).backward()

        if global_step % GRAD_ACCUM_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        # logging & checkpointing
        if global_step % LOG_EVERY == 0:
            print(f"[Epoch {epoch}] step {global_step:>6}  loss = {
                  loss.item()*GRAD_ACCUM_STEPS:.4f}")
            # save a checkpoint
            ckpt_dir = os.path.join(OUTPUT_DIR, f"checkpoint-{global_step}")
            os.makedirs(ckpt_dir, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            # run generation‐based eval
            run_eval(global_step)
