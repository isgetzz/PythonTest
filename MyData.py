# datasets 微模型训练

# #下载数据到本地
# dataset = load_dataset('lansinuote/ChnSentiCorp', cache_dir=r"model/uer/ChnSentiCorp")
# 查看数据集信息
# print(dataset)

from datasets import load_from_disk
# 加载数据集 首次从hugging face 加载，下载完成读取本地缓存
# dataset = load_from_disk(r"D:\PythonTest\model\uer\ChnSentiCorp\lansinuote___chn_senti_corp\default\0.0.0\b0c4c119c3fb33b8e735969202ef9ad13d717e5a")
# 查看数据集信息
# dataset = load_dataset(
#     r"D:\PythonTest\model\uer\ChnSentiCorp\lansinuote___chn_senti_corp\default\0.0.0\b0c4c119c3fb33b8e735969202ef9ad13d717e5a")
# #可以首次加载hugging face 的时候设置
# dataset.save_to_disk(r"data\ChnSentiCorp")
# 直接使用刚才存储的本地文件
# data_train = load_from_disk(r"data\ChnSentiCorp")
# for data in data_train["train"]:
#   print(data)
# print(data_train)
# 数据处理
from torch.utils.data import Dataset
from huggingface_hub import try_to_load_from_cache
class MyDataset(Dataset):
    def __init__(self, spilt):
        self.dataset = load_from_disk(r"D:\AsApp\PythonTest\data\ChnSentiCorp")
        if spilt == "train":
            self.dataset = self.dataset["train"]
        elif spilt == "test":
            self.dataset = self.dataset["test"]
        elif spilt == "validation":
            self.dataset = self.dataset["validation"]
        else:
            print("Split must be 'train' or 'test' or 'validation'")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        text = self.dataset[item]["text"]
        label = self.dataset[item]["label"]
        return text, label


if __name__ == '__main__':
    dataset = MyDataset("train")
    for data in dataset:
        print(data)
