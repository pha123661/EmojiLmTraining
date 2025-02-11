import asyncio
import json

import jsonlines
import tqdm
from datasets import load_dataset
from googletrans import LANGUAGES, Translator
from tqdm import tqdm

ds = load_dataset("KomeijiForce/Text2Emoji")


async def translate(input_text_list):
    async with Translator() as translator:
        translations = await translator.translate(input_text_list, src='en', dest='zh-tw')
        return [translation.text for translation in translations]


async def main():
    # group 5 samples together
    # Output results to emoji_dataset/text2emoji_train.jsonl
    group_size = 5
    output_file = "emoji_dataset/text2emoji_train.jsonl"

    # Resume if the output file already exists
    try:
        with jsonlines.open(output_file, 'r') as reader:
            start_idx = len(list(reader))
        print(f"Resuming from #{start_idx}")
    except FileNotFoundError:
        start_idx = 0

    output_file_ptr = open(output_file, 'a', encoding='utf-8')
    for i in tqdm(range(start_idx, len(ds['train']), group_size)):
        input_text_list = ds['train']['text'][i:i + group_size]
        emoji_list = ds['train']['emoji'][i:i + group_size]
        translated_text_list = None
        while translated_text_list is None:
            try:
                translated_text_list = await translate(input_text_list)
            except Exception as e:
                print(f"Error: {e}")
                continue

        for translated_text, emoji_list in zip(translated_text_list, emoji_list):
            output_file_ptr.write(json.dumps({
                "input": translated_text,
                "output": emoji_list,
            }, ensure_ascii=False) + "\n")

asyncio.run(main())
