# #将模型和分间工具下载到本地，并指定保存路径
# model_name ="uer/gpt2-chinese-cluecorpussmall"
# cache_dir ="model/uer/gpt2-chinese-cluecorpussmall"
# #下载模型
# AutoModelForCausalLM.from_pretrained(model_name,cache_dir=cache_dir)
# #下载分词工具
# # AutoTokenizer.from_pretrained(model_name,cache dir=cache_dir)
# print(f"模型分词器已下载到:{cache_dir}")

# 使用本地下载完成的模型
# dir = r"D:\PythonTest\model\uer\gpt2-chinese-cluecorpussmall\models--uer--gpt2-chinese-cluecorpussmall\snapshots\c2c0249d8a2731f269414cc3b22dff021f8e07a3"
# model = AutoModelForCausalLM.from_pretrained(dir)
# tokenizer = AutoTokenizer.from_pretrained(dir)
# generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
# output = generator("你好啊，我是一个AI模型", max_new_tokens=100, truncation=True, temperature=0.7, top_k=50, top_p=0.9,
# num_return_sequences=1)
# print(output)
