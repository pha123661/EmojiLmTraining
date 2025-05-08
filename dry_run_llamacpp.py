from llama_cpp import Llama

emojilm = Llama(model_path="emojilm-0.6b-f16.gguf")

prompt = "哈哈狗"

# Define the parameters
model_output = emojilm(
    prompt,
    max_tokens=3,
    temperature=0.5,
    top_p=0.8,
    echo=False,
    stop=["\n", "\t", " ", "。", "！", "？"],
    repeat_penalty=1.2,
)
print(model_output)

print(model_output['choices'][0]['text'])
