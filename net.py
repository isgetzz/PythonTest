import torch
from transformers import BertModel

# 设置模型训练设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pretrained = BertModel.from_pretrained("bert-base-chinese").to(DEVICE)
print(f"net: {DEVICE}")


# 定义下游任务模型(将数据进行分类)
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # 上游任务不参与训练
        with torch.no_grad():
            out = pretrained(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # 下游任务参与训练
        out = self.fc(out.last_hidden_state[:, 0])
        out = out.softmax(dim=1)
        return out
