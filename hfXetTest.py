import os
import sys

from huggingface_hub import try_to_load_from_cache

from hugging_face_token import Token

# 官方推荐模型下载加速插件
# 确保环境变量已设置
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# 检查模块是否加载
if "hf_xet" not in sys.modules:
    try:
        import hf_xet

        print("✅ hf_xet 手动导入成功")
    except ImportError:
        print("❌ hf_xet 未安装")

# 高级检测
print("\n=== 环境检测 ===")
print(f"HF_HUB_ENABLE_HF_TRANSFER: {os.getenv('HF_HUB_ENABLE_HF_TRANSFER')}")
print(f"Python路径: {sys.path}")

try:
    # 核心检测方法
    from huggingface_hub._xet import XetFileSystem  # 新版本路径

    print("✅ XetFileSystem 检测成功")
except ImportError:
    try:
        from hf_xet import XetFileSystem  # 旧版本路径

        print("✅ XetFileSystem (旧版) 检测成功")
    except ImportError:
        print("❌ XetFileSystem 未找到")

# 替代检测
print("\n=== 协议支持检测 ===")
print(f"支持协议: {try_to_load_from_cache.__globals__.get('_SUPPORTED_PROTOCOLS', [])}")
print(f"Xet enabled: {'xet' in try_to_load_from_cache.__globals__.get('_SUPPORTED_PROTOCOLS', [])}")
from hf_xet import download_files
from huggingface_hub import list_lfs_files

# 查看模型仓库所有文件
files = list_lfs_files(repo_id="uer/gpt2-chinese-cluecorpussmall", token=Token)

# 打印文件路径和元数据
for file in files:
    print(f"路径: {file}")
    print(f"大小: {file}")
    print(f"类型: {file}")  # file/directory
    print("------")
download_files(files=files, repo_id="uer/gpt2-chinese-cluecorpussmall", local_path="./model")
