import csv
import sys

import jsonlines

dataset_jsonl = sys.argv[1]

range_list = [
    ["1f170", "1f251"],
    ["1f300", "1f64f"],
    ["1f680", "1f6c5"],
    ["2702", "27b0"],
]
emoji_unicode = [int("1f004", 16), int("1f0cf", 16), int("24c2", 16)]
for a, b in range_list:
    emoji_unicode.extend(list(range(int(a, 16), int(b, 16) + 1)))


in_content = []
out_content = []
with open(dataset_jsonl, "r") as f:
    for data in jsonlines.Reader(f):
        # print(data)
        if data["Type"] == 1:
            choose = False
            for c in data["Content"]:
                if ord(c) in emoji_unicode:
                    # print(c)
                    choose = True
                    break
            if choose:
                in_content.append(
                    data["Content"]
                    .replace("\u2009", " ")
                    .replace("\u200c", "")
                    .replace("\u200d", "")
                    .replace("\ufe0f", "")
                    .replace("\n", " ")
                )
            else:
                out_content.append(data["Content"])

with open("emoji_dataset/dataset.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["emoji 版本", "原文（if any）"])
    for data in in_content:
        writer.writerow([data])
