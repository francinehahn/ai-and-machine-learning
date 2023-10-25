from transformers import pipeline

generator = pipeline("text-generation", model="pierreguillou/gpt2-small-portuguese")
text = "Em sentido estrito, ciência refere-se ao sistema de adquirir conhecimento baseado no método científico."

result = generator(text, max_length=100, do_sample=True) #do sample True makes the text more creative
print(result)

