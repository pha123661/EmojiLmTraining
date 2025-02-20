# Emoji Language Model Training

This repository contains scripts and datasets for training an emoji language model.

## Directory Structure

- `dataset_scripts/`: Contains scripts for preprocessing the training dataset.
- `emoji_dataset/`: Directory where the emoji datasets are stored.
- `emoji_data`: Directory where the emoji data are stored.
-
## Setup

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/EmojiLmTraining.git
    cd EmojiLmTraining
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

3. Download `EmojiLM 訓練資料` from drive. Rename it to `emoji_dataset.csv` and place it in the `emoji_dataset/` directory.

## Preprocessing

Step 1. dump copypasta from MongoDB
```sh
python dataset_scripts/dumpcopypasta_jsonl_to_csv.py
```

Step 2. Translate "KomeijiForce/Text2Emoji" to Chinese
```sh
python dataset_scripts/translate_text2emoji_dataset.py
```

Step 3. Preprocess the training dataset
```sh
python dataset_scripts/preprocess_training_dataset.py
```

## Training

```sh
python train_lora.py
```