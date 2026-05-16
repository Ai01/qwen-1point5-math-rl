import os
import re
from datasets import load_dataset, load_from_disk

# --- 配置中心 ---
DATA_PATHS = {
    "gsm8k": "./data/gsm8k",
    "gsm1k": "./data/gsm1k"
}

HF_REPO_IDS = {
    "gsm8k": ("openai/gsm8k", "main"),
    "gsm1k": ("scale-ai/gsm1k", "main")
}

# --- 内部工具函数 ---
def _extract_hash_answer(text):
    """
    从 GSM8K 格式的回答中提取纯数字答案
    例如: "Janet sells ... #### 18" -> "18"
    同时处理逗号，如 "130,000" -> "130000"
    """
    if "####" in text:
        ans = text.split("####")[-1].strip()
    else:
        ans = text.strip()
    
    # 移除逗号和其他可能的非数字干扰字符（保留负号和小数点）
    ans = ans.replace(",", "")
    # 只提取数字部分
    match = re.search(r"[-+]?\d*\.\d+|\d+", ans)
    if match:
        return match.group()
    return ans

# --- 核心下载逻辑 ---
def download_dataset_to_local(dataset_name):
    """
    统一的数据集下载入口
    """
    save_dir = DATA_PATHS.get(dataset_name)
    repo_info = HF_REPO_IDS.get(dataset_name)
    
    if not save_dir or not repo_info:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    print(f"Downloading {dataset_name} to {save_dir}...")
    try:
        # 尝试从指定 Repo 下载
        dataset = load_dataset(*repo_info)
    except Exception as e:
        print(f"Primary source failed for {dataset_name}: {e}")
        if dataset_name == "gsm1k":
            print("Falling back to GSM8K for GSM1k placeholder...")
            dataset = load_dataset(*HF_REPO_IDS["gsm8k"])
        else:
            raise e
            
    dataset.save_to_disk(save_dir)
    print(f"{dataset_name} download completed.")

def ensure_dataset_available(dataset_name):
    """
    确保数据集在本地可用，如果不在则下载
    """
    save_dir = DATA_PATHS.get(dataset_name)
    if not os.path.exists(save_dir):
        download_dataset_to_local(dataset_name)
    return save_dir

# --- 核心加载接口 ---
def get_math_questions(dataset_name="gsm8k", split="train"):
    """
    高内聚的加载接口：自动处理下载、本地加载和格式化
    """
    local_path = ensure_dataset_available(dataset_name)
    
    print(f"Loading {dataset_name} ({split}) from {local_path}")
    dataset_dict = load_from_disk(local_path)
    
    # GSM1k 只有 test set，如果请求 train 则返回 test
    if dataset_name == "gsm1k":
        ds = dataset_dict["test"]
    else:
        ds = dataset_dict[split]
    
    def format_example(example):
        return {
            "prompt": [
                {"role": "system", "content": "You are a helpful assistant that solves math problems step by step. Please enclose your thinking process in <think> tags and provide the final answer at the end."},
                {"role": "user", "content": example["question"]}
            ],
            "answer": _extract_hash_answer(str(example["answer"]))
        }
    
    return ds.map(format_example)

# --- 答案提取工具（供其他模块调用） ---
def extract_numerical_answer(text):
    """
    从模型生成的长文本中提取最后的数值结果
    """
    if "</think>" in text:
        text = text.split("</think>")[-1]
    
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            return None
    return None

if __name__ == "__main__":
    # 测试重构后的逻辑
    for name in ["gsm8k", "gsm1k"]:
        ds = get_math_questions(name, split="test")
        print(f"\nDataset: {name}")
        print(f"Size: {len(ds)}")
        print(f"Example answer: {ds[0]['answer']}")
