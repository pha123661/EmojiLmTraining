import argparse
import os

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class EmojiLM:
    def __init__(self, model_name, lora_path) -> None:
        self.device = 'cuda'
        self.task_prefix = "emoji: "
        self.prepare_model(model_name, lora_path)
        text_input = "那你很厲害誒"
        print("開機測試：", text_input, self.serve(text_input))

    def prepare_model(self, model_name, lora_path):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=False)
        self.tokenizer.truncation_side = 'left'

        # peft_config = PeftConfig.from_pretrained(lora_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model = PeftModel.from_pretrained(model, lora_path)
        self.model = self.model.merge_and_unload()
        self.model.eval()
        self.model.to(self.device)

    @torch.no_grad()
    def serve(self, inputs):
        inputs = [self.task_prefix + inputs]
        model_inputs = self.tokenizer(
            inputs, max_length=128, padding='do_not_pad', truncation=True, return_tensors="pt")
        outputs = self.model.generate(
            input_ids=model_inputs["input_ids"].to(self.device), max_new_tokens=5, do_sample=False)

        ret = self.tokenizer.batch_decode(
            outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]
        return ret

    def push_to_hub(self, hub_name):
        self.model.push_to_hub(hub_name, private=False)
        self.tokenizer.push_to_hub(hub_name, private=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Run PEFT inference")
    parser.add_argument("--model", type=str, default="google/mt5-base")
    parser.add_argument("--lora", type=str, required=True)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--hub_name", type=str, default="EmojiLMSeq2SeqLoRA", help="Model name for pushing to hub")
    return parser.parse_args()


def main():
    args = parse_args()
    EmojiLmWorker = EmojiLM(args.model, args.lora)
    if args.upload:
        EmojiLmWorker.push_to_hub(args.hub_name)
        from datasets import load_dataset
        dataset = load_dataset('./emoji_dataset')
        dataset.push_to_hub("EmojiAppendDataset", private=True)
    while True:
        try:
            inp = input("Enter input: ")
        except (KeyboardInterrupt, EOFError):
            break
        print(inp, EmojiLmWorker.serve(inp))


if __name__ == "__main__":
    main()
