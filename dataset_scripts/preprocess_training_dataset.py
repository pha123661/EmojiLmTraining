import argparse
import csv
import os
import random
import re
from collections import Counter

import jsonlines
from tqdm.auto import tqdm

TEXT2EMOJIPROPOTION = 0.5

random.seed(11944004)

parser = argparse.ArgumentParser()
parser.add_argument("--emoji_dataset", type=str,
                    default="emoji_dataset/emoji_dataset.csv")
parser.add_argument("--text2emoji_train", type=str,
                    default="emoji_dataset/text2emoji_train.jsonl")
args = parser.parse_args()

with open(args.emoji_dataset) as f:
    reader = csv.reader(f)
    emoji_dataset_list = [row[0] for row in reader]


def normalize_whitespace(text):
    # Remove thin space and zero width space
    text = re.sub(r"[\u2009\n]", " ", text)
    text = re.sub(r"[\u200c\u200d\ufe0f]", "", text)

    # Remove replacement character
    text = re.sub(r"\ufffd", "", text)
    return text


# Load selected emojis and rejected emojis
with open("emoji_data/selected_emojis.txt", "r") as f:
    selected_emojis_list = f.readline()
emoji_pattern = f"([^{selected_emojis_list}]+)([{selected_emojis_list}|\n]+)"

with open("emoji_data/rejected_emojis.txt", "r") as f:
    reject_list = f.read().splitlines()
reject_pattern = '[' + ''.join(reject_list) + ']'


def extract_continuous_emojis(text):
    matches = re.findall(emoji_pattern, text)
    return matches


def postprocess(text):
    text = re.sub(reject_pattern, '', text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"^[\s,。、，]+|[\s,。、，]+$", "", text)
    return text


def contains_three_continuous_chars(sentence):
    # Pattern to match three continuous characters
    three_continuous_chars_pattern = r"(.)\1\1"
    match = re.search(three_continuous_chars_pattern, sentence)
    return bool(match)


def contains_only_ascii(s):
    return bool(re.match('^[\x00-\x7F]+$', s))


dataset = []
for text in tqdm(emoji_dataset_list):
    text = normalize_whitespace(text)
    extracted_content = extract_continuous_emojis(text)
    for extracted_test, extracted_emojis in extracted_content:
        extracted_test = postprocess(extracted_test)
        extracted_emojis = postprocess(extracted_emojis)
        if contains_three_continuous_chars(extracted_test):
            continue
        if len(extracted_test) == 0 or len(extracted_emojis) == 0:
            continue
        dataset.append({"input": extracted_test, "output": extracted_emojis})

# Text2EMoji Dataset
# Randomly sample the same number of lines as the text2emoji dataset


def reservoir_sample(file_path, sample_size):
    reservoir = []
    with jsonlines.open(file_path, 'r') as reader:
        for i, line in enumerate(reader):
            if i < sample_size:
                reservoir.append(line)
            else:
                j = random.randint(0, i)
                if j < sample_size:
                    reservoir[j] = line
    return reservoir


text2emoji_dataset_list = reservoir_sample(
    args.text2emoji_train, TEXT2EMOJIPROPOTION * len(dataset))

for sample in tqdm(text2emoji_dataset_list):
    text = postprocess(normalize_whitespace(sample['input']))
    extracted_emoji = postprocess(sample['output'])
    if contains_three_continuous_chars(text):
        continue
    if len(text) == 0 or len(extracted_emoji) == 0:
        continue
    dataset.append({"input": text, "output": extracted_emoji})

print("##########")
print(f"Total {len(dataset)} lines after preprocessing")

# Remove consecutive lines with the same output of more than 3
prev_output = ""
prev_count = 0
dataset_without_consecutives = []
for data in dataset:
    if data["output"] == prev_output:
        prev_count += 1
    else:
        prev_count = 0
    if prev_count < 3:
        dataset_without_consecutives.append(data)
    prev_output = data["output"]
dataset = dataset_without_consecutives

print("##########")
print(f"Total {len(dataset)} lines after consecutive removal")

# Remove duplicate inputs if the output appears many times
emoji_counter = Counter()
for text in dataset:
    emoji_counter.update(text['output'])

unique_dataset = []
seen = set()
for data in dataset:
    if (data['input'], data['output']) not in seen:
        unique_dataset.append(data)
        seen.add((data['input'], data['output']))
    else:
        if sum(emoji_counter[c] for c in data['output']) < 3 * len(data['output']):
            unique_dataset.append(data)
dataset = unique_dataset

print("##########")
print(f"Total {len(dataset)} lines after duplicate removal")

print("##########")
print(
    f"Samples with longer than 128 input chars: {len([d for d in dataset if len(d['input']) > 128])}")
print(
    f"Samples with >6 output chars: {len([d for d in dataset if len(d['output']) > 6])}")
print("##########")

with jsonlines.open("emoji_dataset/train_and_val.jsonl", "w") as writer:
    writer.write_all(dataset)

emoji_counter = Counter()
for text in dataset:
    emoji_counter.update(text['output'])
print(emoji_counter)


def split_jsonl(input_file, train_file, val_file, split_ratio=0.8):
    if split_ratio <= 0 or split_ratio >= 1:
        raise ValueError("Split ratio should be between 0 and 1")

    with jsonlines.open(input_file, 'r') as reader:
        data = list(reader)
        random.shuffle(data)

        split_index = int(len(data) * split_ratio)
        train_data = data[:split_index]
        val_data = data[split_index:]

    # Writing to train.jsonl
    with jsonlines.open(train_file, 'w') as writer_train:
        for item in train_data:
            writer_train.write(item)

    # Writing to val.jsonl
    with jsonlines.open(val_file, 'w') as writer_val:
        for item in val_data:
            writer_val.write(item)


split_jsonl("emoji_dataset/train_and_val.jsonl", "emoji_dataset/train.jsonl",
            "emoji_dataset/val.jsonl", split_ratio=0.95)

# print("Plotting...")
# input_lengths = [len(d['input']) for d in dataset]
# output_lengths = [len(d['output']) for d in dataset]
# sns.set_theme(style="whitegrid")

# fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# # Plot input length distribution on a log scale
# sns.histplot(input_lengths, bins=100, ax=axes[0])
# axes[0].set_title('Input Length Distribution')
# axes[0].set_xlabel('Length')
# axes[0].set_ylabel('Count')
# axes[0].set_yscale('log')

# # Plot output length distribution on a log scale
# sns.histplot(output_lengths, bins=100, ax=axes[1])
# axes[1].set_title('Output Length Distribution')
# axes[1].set_xlabel('Length')
# axes[1].set_ylabel('Count')
# axes[1].set_yscale('log')

# plt.tight_layout()
# plt.savefig("emoji_dataset/length_distributions.png")

# # Plot emoji frequency
# emoji_counter = Counter()
# for text in dataset:
#     emoji_counter.update(text['output'])
# plt.figure(figsize=(12, 6))

# values = list(emoji_counter.values())
# sorted_idx = sorted(range(len(values)),
#                     key=lambda k: values[k], reverse=True)
# sns.barplot([values[i] for i in sorted_idx])
# plt.xticks([])
# plt.title('Emoji Frequency Distribution')
# plt.ylabel('Count')
# plt.yscale('log')
# plt.tight_layout()
# plt.savefig("emoji_dataset/emoji_distribution.png")
