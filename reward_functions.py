import re

def accuracy_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """
    Reward for getting the correct numerical answer.
    """
    rewards = []
    for completion, correct_answer in zip(completions, answer):
        # Handle list-style completions (if TRL passes chat format)
        if isinstance(completion, list):
            # Extract content from the last assistant message
            content = completion[-1]["content"] if completion else ""
        else:
            content = completion

        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", content)
        if numbers:
            last_number = numbers[-1]
            if last_number == correct_answer:
                rewards.append(2.0)
            else:
                rewards.append(0.0)
        else:
            rewards.append(0.0)
    return rewards

def format_reward_func(completions, **kwargs) -> list[float]:
    """
    Reward for following the <think>...</think> format.
    """
    rewards = []
    for completion in completions:
        if isinstance(completion, list):
            content = completion[-1]["content"] if completion else ""
        else:
            content = completion

        pattern = r"^<think>.*?</think>\s*.*"
        if re.search(pattern, content, re.DOTALL):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards

def reasoning_steps_reward_func(completions, **kwargs) -> list[float]:
    """
    Encourage more reasoning steps by rewarding longer content inside <think> tags.
    """
    rewards = []
    for completion in completions:
        if isinstance(completion, list):
            content = completion[-1]["content"] if completion else ""
        else:
            content = completion

        think_content = re.findall(r"<think>(.*?)</think>", content, re.DOTALL)
        if think_content:
            length = len(think_content[0].split())
            reward = min(length / 200.0, 0.5) 
            rewards.append(reward)
        else:
            rewards.append(0.0)
    return rewards
