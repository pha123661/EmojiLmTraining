import argparse
import os

from huggingface_hub import HfApi, HfFolder, create_repo, upload_folder
from transformers import AutoModel, AutoTokenizer


def update_tokenizer_chat_template(model_dir, template_path):
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=True)

    if template_path:
        with open(template_path, 'r', encoding='utf-8') as f:
            chat_template = f.read()

        tokenizer.chat_template = chat_template
        tokenizer.save_pretrained(model_dir)
        print(f"Tokenizer updated and saved to {model_dir}")

    # Check if chat_template is correctly set
    chat = [
        {"role": "user", "content": "First user message: Hello, how are you?"},
        {"role": "assistant", "content": "First assistant message: I'm doing great. How can I help you today?"},
        {"role": "user", "content": "Second and last user message: I'd like to show off how chat templating works!"},
    ]

    print(tokenizer.apply_chat_template(chat, tokenize=False))
    return model_dir


def create_and_upload(model_dir, repo_id):
    token = HfFolder.get_token()
    if token is None:
        raise ValueError(
            "❌ Please log in with `huggingface-cli login` before running this script.")

    api = HfApi()

    try:
        api.create_repo(repo_id, exist_ok=True, repo_type="model")
        print(
            f"📁 Repo created or already exists: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"❌ Failed to create repo: {e}")
        return

    upload_folder(
        folder_path=model_dir,
        repo_id=repo_id,
        repo_type="model",
    )
    print(
        f"✅ Model and tokenizer uploaded to https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Update tokenizer chat template and upload to Hugging Face.")
    parser.add_argument("model_dir", help="Path to local Hugging Face model directory.")
    parser.add_argument("--template_path", default="./custom_chat_template_emojilm.jinja",
                        help="Path to new chat template file (.jinja).")
    parser.add_argument("--repo", help="Repo ID on Hugging Face Hub.")

    args = parser.parse_args()

    update_tokenizer_chat_template(
        args.model_dir, args.template_path)
    if args.repo:
        create_and_upload(args.model_dir, args.repo)


if __name__ == "__main__":
    main()
