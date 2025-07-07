# 训练模型

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from MyData import MyDataset
from net import Model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 分词器
token = BertTokenizer.from_pretrained('bert-base-chinese')
print(f"train: {DEVICE}")


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
    print("collate_fn", sente, input_ids)
    return input_ids, attention_mask, token_type_ids, labels


# 创建数据集
train_dataset = MyDataset("train")
print("train_dataset：", train_dataset.dataset)
# 数据加载器 dataset 数据源 batch_size 每组的长度 collate_fn collate_fn处理内容编码格式
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, drop_last=True, collate_fn=collate_fn)
if __name__ == '__main__':
    # 模型放入GPU
    model = Model().to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=5e-4)
    # 分类任务 用于量化模型预测与真实标签之间的差异
    loss_func = torch.nn.CrossEntropyLoss()
    model.train()
    # train_loader 使用 enumerate 迭代器遍历  collate_fn 方法定义返回的参数
    for epoch in range(1):
        for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(train_loader):
            # 将张量从当前设备（默认是 CPU）移动到 DEVICE（如 GPU , 如果 DEVICE 已经是当前设备，操作不会产生实际拷贝（返回原张量）
            input_ids, attention_mask, token_type_ids, labels = input_ids.to(DEVICE), \
                attention_mask.to(DEVICE), token_type_ids.to(DEVICE), labels.to(DEVICE)
            # 把collate_fn编码返回的所有的张量练入下游训练模型才能开始训练
            out = model(input_ids, attention_mask, token_type_ids)
            # 计算丢失精度  原数据格式里的labels值跟模型计算集合值做对比
            loss = loss_func(out, labels)
            # 比默认的 set_to_zero 更
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if i % 5 == 0:
                out = out.argmax(dim=1)
                # 计算准确率
                acc = (out == labels).sum().item() / len(labels)
                print("训练数据中：", epoch, i, loss.item(), acc)
        # print(epoch, i, loss.item(), (out == labels).sum().item() / len(labels))
        # 把训练完成的数据保存，因为 batch_encode_plus 使用的pt所以接收也需要保持一致
        torch.save(model.state_dict(), f"params/{epoch}bert.pt")
        print("参数保存成功", epoch)
