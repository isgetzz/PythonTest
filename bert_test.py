# 测试训练完成的模型

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer

import MyData
from net import Model

# 设置计算的类型 GPU或者CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 分词器
token = BertTokenizer.from_pretrained("bert-base-chinese")
# 数据类型test、train、validation
test_dataset = MyData.MyDataset("test")


# 自定义函数对数据进行编码处理
# data是batch_size长度的源数据集合
# 每个item的数据体跟据MyDataset定义的
def collate_fn(data):
    sente = [i[0] for i in data]
    label = [i[1] for i in data]
    # padding=True自动填充到最长序列  truncation=True,  # 自动截断到最大长度 max_length=128,  # 最大长度限制  return_tensors="pt",
    # 返回 PyTorch 张量（可选 "tf" 或 "np" return_length=True,  # 返回每个样本的实际长度（非填充长度）
    data = token.batch_encode_plus(batch_text_or_text_pairs=sente, truncation=True, padding="max_length",
                                   max_length=500, return_tensors="pt", return_length=True)
    # sente 调用batch_encode_plus编码完生成一个分词器集合
    # input_ids 将文本转换为 token 的 ID 序列  [CLS]（101）：BERT 的开头特殊 token [SEP]（102）：BERT 的结尾特殊 token（如果是句子对，第二个句子后也会加 [SEP]）
    # attention_mask 标识哪些 token 是真实数据（1），哪些是填充的（0）
    # token_type_ids 区分句子对中的不同句子（单句时全为 0） 单句输入：全 0（如 [0, 0, 0, 0, 0, 0, 0, 0]） 句子对（如问答任务）：第一句 0，第二句 1
    # length 返回每个样本的实际 token 长度（不包括填充） ["Hello", "Hi there"] → [3, 4]
    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"]
    token_type_ids = data["token_type_ids"]
    length = data["length"]
    labels = torch.LongTensor(label)
    print("collate_fn", sente)
    return input_ids, attention_mask, token_type_ids, labels


loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=True, drop_last=True, collate_fn=collate_fn)
if __name__ == '__main__':
    acc = 0
    total = 0
    model = Model().to(DEVICE)
    model.load_state_dict(torch.load(r"params\0bert.pt"))
    model.eval()

    for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(loader):
        # 将张量从当前设备（默认是 CPU）移动到 DEVICE（如 GPU , 如果 DEVICE 已经是当前设备，操作不会产生实际拷贝（返回原张量）
        input_ids, attention_mask, token_type_ids, labels = input_ids.to(DEVICE), attention_mask.to(
            DEVICE), token_type_ids.to(DEVICE), labels.to(DEVICE)
        # 得移动所有的张量模型能开始训练
        out = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        out = out.argmax(dim=1)
        acc += (out == labels).sum().item()
        total += len(labels)
        print("计算中", i, (out == labels).sum().item() / len(labels))
    print("测试精度", acc / total)
