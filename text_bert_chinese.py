# 底层实现
from transformers import BertTokenizer

token = BertTokenizer.from_pretrained("bert-base-chinese")
vocab = token.get_vocab()
add = token.add_tokens(new_tokens=["阳光", "大地"])
token.add_special_tokens({"eos_token": "[E0S]"})
# text = token.encode(text="我是一个小太阳，温暖整个世界！", truncation=True, add_special_tokens=True, return_tensors=None)
# print(token.decode(text))
print(vocab)
print("阳光" in token.vocab)
print(add)
