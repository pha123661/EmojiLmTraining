from transformers import pipeline

p = pipeline("text2text-generation",
             model="liswei/EmojiLMSeq2SeqLoRA",)

text = p("那你很厲害誒")
print(text)
