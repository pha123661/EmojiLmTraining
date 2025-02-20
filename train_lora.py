import os
import random

import jsonlines
import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                          DataCollatorForSeq2Seq, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments)


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


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = \
        os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(",")[0]
    dataset_name = "./emoji_dataset"
    model_name = "google/mt5-base"
    output_dir = "EmojiLMSeq2SeqLoRA"

    task_prefix = ""
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

        model_inputs = tokenizer(
            inputs, max_length=max_length, padding='do_not_pad', truncation=True)
        labels = tokenizer(
            text_target=targets, max_length=max_length, padding='do_not_pad', truncation=True)

        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]
        model_inputs["labels"] = labels["input_ids"]

        if len(model_inputs["labels"]) != len(model_inputs["input_ids"]):
            raise ValueError("The length of labels and inputs must match.")

        return model_inputs

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
        task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=16, lora_alpha=32, lora_dropout=0.1
    )

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
        dataloader_num_workers=0 if torch.backends.mps.is_available() else 4,
        generation_max_length=5,
        output_dir=output_dir,
        predict_with_generate=True,
        eval_strategy="epoch",
        logging_steps=100,
        save_strategy='epoch',
        overwrite_output_dir=True,
        include_for_metrics=['inputs'],
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
