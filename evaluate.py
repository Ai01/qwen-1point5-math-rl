import torch
import re
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from data_utils import get_math_questions, extract_numerical_answer
import os
import json
from datetime import datetime
from tqdm import tqdm

def calculate_score(model_answer, correct_answer):
    """
    评分逻辑：和正确回答的偏差越大分数越低。
    exact match = 1.0
    Score = 1 / (1 + abs(model - correct))
    """
    if model_answer is None:
        return 0.0
    
    try:
        # 统一处理数据格式
        from data_utils import _extract_hash_answer
        correct_val_str = _extract_hash_answer(str(correct_answer))
        correct_val = float(correct_val_str.replace(",", ""))
        
        diff = abs(model_answer - correct_val)
        if diff == 0:
            return 1.0
        return 1.0 / (1.0 + diff)
    except Exception as e:
        return 0.0

def generate_response(model, tokenizer, prompt_messages):
    """
    让模型生成回答
    """
    input_ids = tokenizer.apply_chat_template(
        prompt_messages, 
        tokenize=True, 
        add_generation_prompt=True, 
        return_tensors="pt"
    ).to(model.device)
    
    if hasattr(input_ids, "input_ids"):
        input_ids = input_ids.input_ids

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids, 
            max_new_tokens=512, 
            do_sample=False, 
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
    return response

def run_inference_and_save(model, tokenizer, dataset, model_name, result_dir):
    """
    运行推理并将结果保存到 JSON
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_{timestamp}.json"
    save_path = os.path.join(result_dir, filename)
    
    results = []
    print(f"Running inference for {model_name}...")
    
    for i, example in enumerate(tqdm(dataset)):
        response = generate_response(model, tokenizer, example["prompt"])
        results.append({
            "id": i,
            "input": example["prompt"],
            "target": example["answer"],
            "output": response
        })
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    return save_path

def plot_from_results(before_json, after_json, score_dir):
    """
    从保存的 JSON 结果中计算评分并绘图
    """
    with open(before_json, 'r') as f:
        before_data = json.load(f)
    with open(after_json, 'r') as f:
        after_data = json.load(f)
        
    before_scores = []
    after_scores = []
    
    for b, a in zip(before_data, after_data):
        # 计算训练前得分
        b_val = extract_numerical_answer(b["output"])
        before_scores.append(calculate_score(b_val, b["target"]))
        
        # 计算训练后得分
        a_val = extract_numerical_answer(a["output"])
        after_scores.append(calculate_score(a_val, a["target"]))
    
    # 绘图
    plt.figure(figsize=(15, 7))
    plt.plot(before_scores, label='Before Training', alpha=0.6, color='#ff9999', marker='o', markersize=2, linestyle='')
    plt.plot(after_scores, label='After Training', alpha=0.6, color='#66b3ff', marker='x', markersize=2, linestyle='')
    
    plt.axhline(y=np.mean(before_scores), color='r', linestyle='--', label=f'Avg Before ({np.mean(before_scores):.3f})')
    plt.axhline(y=np.mean(after_scores), color='b', linestyle='--', label=f'Avg After ({np.mean(after_scores):.3f})')
    
    plt.ylabel('Score (1/(1+diff))')
    plt.xlabel('Sample Index')
    plt.title(f'Performance Comparison (Total Samples: {len(before_scores)})')
    plt.legend()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(score_dir, f"score_comparison_{timestamp}.png")
    plt.savefig(plot_path)
    print(f"Score plot saved to: {plot_path}")
    return plot_path

def main():
    # 路径配置
    MODEL_ID = "/home/bhh/Desktop/ai/MathRL/models/qwen/Qwen2___5-1___5B-Instruct"
    ADAPTER_PATH = "qwen-grpo-gsm8k-final"
    RESULT_DIR = "result"
    SCORE_DIR = "result-score"
    
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(SCORE_DIR, exist_ok=True)
    
    # 加载数据
    dataset = get_math_questions("gsm1k", split="test")
    # 如果需要测试全量，直接使用 dataset；如果需要限制数量，可以使用 select
    dataset = dataset.select(range(min(100, len(dataset)))) 

    # 1. 基础模型推理
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
    
    before_json = run_inference_and_save(model, tokenizer, dataset, "base_model", RESULT_DIR)
    
    # 2. 训练后模型推理
    if os.path.exists(ADAPTER_PATH):
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)
        after_json = run_inference_and_save(model, tokenizer, dataset, "trained_model", RESULT_DIR)
        
        # 3. 计算得分并绘图
        plot_from_results(before_json, after_json, SCORE_DIR)
    else:
        print("Adapter not found, skipping post-training evaluation.")

if __name__ == "__main__":
    main()
