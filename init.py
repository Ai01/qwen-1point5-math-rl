import os
import subprocess
import sys
from data_utils import download_dataset_to_local, DATA_PATHS

def run_command(command):
    print(f"Executing: {command}")
    try:
        subprocess.check_call(command, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")

def init_environment():
    print("=== MathRL Environment Initialization ===\n")

    # 1. 创建必要目录
    dirs = ["data", "result", "result-score", "models"]
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"Created directory: {d}")
        else:
            print(f"Directory already exists: {d}")

    # 2. 安装依赖
    print("\n--- Checking Dependencies ---")
    if os.path.exists("requirements.txt"):
        # 使用当前 python 解释器对应的 pip
        pip_cmd = f"{sys.executable} -m pip install -r requirements.txt"
        run_command(pip_cmd)
    else:
        print("requirements.txt not found, skipping dependency installation.")

    # 3. 下载数据集
    print("\n--- Preparing Datasets ---")
    for dataset_name in DATA_PATHS.keys():
        save_dir = DATA_PATHS[dataset_name]
        if not os.path.exists(save_dir):
            try:
                download_dataset_to_local(dataset_name)
            except Exception as e:
                print(f"Failed to download {dataset_name}: {e}")
        else:
            print(f"Dataset {dataset_name} already exists at {save_dir}")

    # 4. 下载基础模型
    print("\n--- Preparing Base Model ---")
    base_model_name = 'qwen/Qwen2.5-1.5B-Instruct'
    local_model_root = './models'
    
    # 检查模型是否已存在 (简单判断目录是否存在)
    # ModelScope 的下载路径通常是 local_model_root/qwen/Qwen2.5-1.5B-Instruct
    expected_path = os.path.join(local_model_root, 'qwen', 'Qwen2.5-1.5B-Instruct')
    
    if not os.path.exists(expected_path):
        print(f"Downloading base model {base_model_name} to {local_model_root}...")
        try:
            from modelscope import snapshot_download
            model_dir = snapshot_download(base_model_name, cache_dir=local_model_root)
            print(f"Model downloaded to: {model_dir}")
        except ImportError:
            print("Error: 'modelscope' not installed. Please ensure dependencies are installed first.")
        except Exception as e:
            print(f"Failed to download model: {e}")
    else:
        print(f"Base model already exists at {expected_path}")

    print("\n=== Initialization Complete ===")
    print("You can now start training with: python3 train_grpo.py")

if __name__ == "__main__":
    init_environment()
