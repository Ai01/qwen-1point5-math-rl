import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import GRPOTrainer, GRPOConfig
from data_utils import get_math_questions
from reward_functions import accuracy_reward_func, format_reward_func, reasoning_steps_reward_func

def train():
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    
    # 1. Load Dataset
    dataset = get_math_questions("gsm8k", split="train")
    
    # 2. Training Config
    training_args = GRPOConfig(
        output_dir="qwen-grpo-gsm8k",
        learning_rate=5e-6,
        per_device_train_batch_size=4, # Prompts per device
        gradient_accumulation_steps=1, 
        max_completion_length=512,
        num_generations=4, # Generations per prompt
        max_steps=1,
        save_steps=1,
        logging_steps=1,
        bf16=True,
        report_to="none",
        use_vllm=False,
    )
    
    # 3. LoRA Config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # 4. Initialize Trainer
    trainer = GRPOTrainer(
        model=model_id,
        reward_funcs=[accuracy_reward_func, format_reward_func, reasoning_steps_reward_func],
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
    )
    
    # 5. Start Training
    print("Starting GRPO Training...")
    trainer.train()
    
    # 6. Save the model
    trainer.save_model("qwen-grpo-gsm8k-final")
    print("Training completed and model saved.")

if __name__ == "__main__":
    train()
