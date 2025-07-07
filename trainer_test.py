# 手动测试训练好的模型
import torch
from transformers import BertTokenizer

from net import Model

# 定义计算模型的硬件类型,GPU 或者 cpu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 根据源数据labels 定义的0 差评 1好评
names = ["差评", "好评"]
model = Model().to(DEVICE)
# 分词器
token = BertTokenizer.from_pretrained("bert-base-chinese")


# 自定义函数对数据进行编码处理
# data==输入值
def collate_fn(data):
    # padding=True自动填充到最长序列  truncation=True,  # 自动截断到最大长度 max_length=128,  # 最大长度限制  return_tensors="pt",
    # 返回 PyTorch 张量（可选 "tf" 或 "np" return_length=True,  # 返回每个样本的实际长度（非填充长度）
    data = token.batch_encode_plus(batch_text_or_text_pairs=[data], truncation=True, padding="max_length",
                                   max_length=500, return_tensors="pt", return_length=True)
    # sente 调用batch_encode_plus编码完生成一个分词器集合
    # input_ids 将文本转换为 token 的 ID 序列  [CLS]（101）：BERT 的开头特殊 token [SEP]（102）：BERT 的结尾特殊 token（如果是句子对，第二个句子后也会加 [SEP]）
    # attention_mask 标识哪些 token 是真实数据（1），哪些是填充的（0）
    # token_type_ids 区分句子对中的不同句子（单句时全为 0） 单句输入：全 0（如 [0, 0, 0, 0, 0, 0, 0, 0]） 句子对（如问答任务）：第一句 0，第二句 1
    # length 返回每个样本的实际 token 长度（不包括填充） ["Hello", "Hi there"] → [3, 4]
    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"]
    token_type_ids = data["token_type_ids"]
    return input_ids, attention_mask, token_type_ids


def result():
    # 加载训练完成的模型
    model.load_state_dict(torch.load(r"params\0bert.pt"))
    # 计算评估
    model.eval()
    while True:
        data = input("请输入你的描述，按q退出！")
        if data == "q":
            break
        # 将张量从当前设备（默认是 CPU）移动到 DEVICE（如 GPU , 如果 DEVICE 已经是当前设备，操作不会产生实际拷贝（返回原张量）
        input_ids, attention_mask, token_type_ids = collate_fn(data)
        input_ids, attention_mask, token_type_ids = input_ids.to(DEVICE), attention_mask.to(DEVICE), token_type_ids.to(
            DEVICE)
        # 不参与训练
        with torch.no_grad():
            out = model(input_ids, attention_mask, token_type_ids)
            out = out.argmax(dim=1)
            print("你的评价是：", names[out])


if __name__ == '__main__':
    result()
