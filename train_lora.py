import os
import random
from argparse import ArgumentParser

import jsonlines
import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (AutoModelForCausalLM, AutoModelForSeq2SeqLM,
                          AutoTokenizer, DataCollatorForSeq2Seq,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments)


class SampleLabelCollator(DataCollatorForSeq2Seq):
    def __call__(self, features, *args, **kwargs):
        labels = [feature["labels"]
                  for feature in features] if "labels" in features[0].keys() else None
        if labels is None:
            return super().__call__(features, *args, **kwargs)

        for i in range(len(labels)):
            label = labels[i]
            emoji_length = len(label) - 2  # BOS and EOS
            if emoji_length > 3:
                # print(label, self.tokenizer.decode(label), end=" -> ")
                sampled = [label[i]
                           for i in sorted(random.sample(range(len(label)), 3))]  # to prevent the same order
                label = [label[0], *sampled, label[-1]]
                # print(label, self.tokenizer.decode(label))
                labels[i] = label
        features = [{"labels": label, **feature}
                    for label, feature in zip(labels, features)]
        return super().__call__(features, *args, **kwargs)

def parse_args():
    parser = ArgumentParser(description="Train LoRA model")
    parser.add_argument(
        "--dataset_name", type=str, default="./emoji_dataset", help="Dataset name"
    )
    parser.add_argument(
        "--model_name", type=str, default="Qwen/Qwen3-0.6B-Base", help="Model name"
    )
    parser.add_argument(
        "--output_dir", type=str, default="QwenEmojiLMSeq2SeqLoRA", help="Output directory"
    )
    parser.add_argument(
        "--task_type", type=str, default="causal-lm", help="Task type", choices=["causal-lm", "seq2seq-lm"]
    )
    return parser.parse_args()

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = \
        os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(",")[0]
    args = parse_args()
    dataset_name = args.dataset_name
    model_name = args.model_name
    output_dir = args.output_dir

    task_prefix = "emoji: "
    max_length = 128

    dataset = load_dataset(dataset_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    tokenizer.truncation_side = 'left'

    def preprocess_function(examples):
        inputs, targets = [], []
        # examples['input'], examples['output']
        for i in range(len(examples['input'])):
            if not (examples['input'][i] and examples['output'][i]):
                raise ValueError(f"Empty input or output, `{examples['input'][i]}` -> `{examples['output'][i]}`")
            inputs.append(task_prefix + examples['input'][i])
            targets.append(examples['output'][i])

        source_text = tokenizer(
            inputs, max_length=max_length, padding='do_not_pad', truncation=True)
        target_text = tokenizer(
            text_target=targets, max_length=max_length, padding='do_not_pad', truncation=True)

        target_text["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in target_text["input_ids"]
        ]

        if args.task_type == "causal-lm":
            input_ids = []
            labels = []
            for src, label in zip(source_text["input_ids"], target_text["input_ids"]):
                src_len = len(src)
                input_ids.append(src + label)
                labels.append([-100] * src_len + label)
        elif args.task_type == "seq2seq-lm":
            input_ids = source_text["input_ids"]
            labels = target_text["input_ids"]

        return dict(input_ids=input_ids, labels=labels)


    dataset = dataset.map(preprocess_function, batched=True)

    # Training
    current_eval_epoch = 0

    def compute_metrics(eval_preds):
        nonlocal current_eval_epoch
        inputs = eval_preds.inputs
        labels = eval_preds.label_ids
        inputs = np.where(inputs != -100, inputs, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_inputs = tokenizer.batch_decode(
            inputs, skip_special_tokens=True)
        decoded_preds = tokenizer.batch_decode(
            eval_preds.predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(
            labels, skip_special_tokens=True)

        output_file = f"{output_dir}/predictions_{current_eval_epoch}.jsonl"
        output_jsonl = []
        for input_text, output_text, gt_text in zip(decoded_inputs, decoded_preds, decoded_labels):
            output_dict = {
                "input": input_text,
                "output": output_text,
                "gt": gt_text
            }
            output_jsonl.append(output_dict)

        with jsonlines.open(output_file, "w") as writer:
            writer.write_all(output_jsonl)
        current_eval_epoch += 1

        return {}

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM if args.task_type == "causal-lm" else TaskType.SEQ2SEQ_LM,
        inference_mode=False, r=16, lora_alpha=32, lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    if args.task_type == "causal-lm":
        model = AutoModelForCausalLM.from_pretrained(model_name)
    elif args.task_type == "seq2seq-lm":
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    data_collator = SampleLabelCollator(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
    )


    training_args = Seq2SeqTrainingArguments(
        # Actual Training
        per_device_train_batch_size=24,
        per_device_eval_batch_size=24,
        learning_rate=1e-4,
        num_train_epochs=150,
        warmup_steps=500,
        label_smoothing_factor=0.1,
        # Data & Saving
        dataloader_num_workers=4 if not torch.backends.mps.is_available() else 0, # see: https://github.com/UKPLab/sentence-transformers/issues/3014
        generation_max_length=5,
        output_dir=output_dir,
        predict_with_generate=True,
        eval_strategy="epoch",
        logging_steps=100,
        save_strategy='epoch',
        overwrite_output_dir=True,
        include_for_metrics=['inputs'],
        label_names=["labels"],
        save_total_limit=10,
        # Tensorboard settings
        report_to=["tensorboard"],
        logging_dir=f"{output_dir}/logs",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()


if __name__ == "__main__":
    main()
