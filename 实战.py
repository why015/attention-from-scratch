from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("bert-base-chinese")
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

inputs = tokenizer("我爱你", return_tensors="pt")
outputs = model(**inputs)
print(outputs.last_hidden_state.shape)