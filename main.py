from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")
text = "Replace me by any text you'd like."
encoded_input = tokenizer.encode(text)
print(encoded_input)

for idx, t in enumerate(tokenizer.ids_to_tokens.values()):
    if "unused" not in t:
        print(f"{idx}: {t}")