from transformers import pipeline

mask = pipeline("fill-mask", model="neuralmind/bert-base-portuguese-cased")
text = "Existe uma chance do corpo cair no [MASK]"
result = mask(text)

for x in range(len(result)):
    print(result[x])