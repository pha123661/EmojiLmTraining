import csv

import emoji

emoji_dict = emoji.unicode_codes.data_dict.EMOJI_DATA
output_csv = []
for k, v in emoji_dict.items():
    output_csv.append([f"{v['en'][1:]} {k}", ""])
    output_csv.append([f"{v['zh'][1:]} {k}", ""])
    if 'aliases' in v:
        for alias in v['aliases']:
            output_csv.append([f"{alias} {k}", ""])

# write to csv

with open('emoji_dataset/zh_cn_emoji_name.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["emoji 版本", "原文（if any）"])
    writer.writerows(output_csv)
