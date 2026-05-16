# MathRL: 基于 GRPO 的数学推理强化学习

本项目旨在通过强化学习（特别是 GRPO 算法）提升大型语言模型（如 Qwen 2.5）的数学逻辑推理能力。通过在 GSM8K 数据集上进行训练，并使用防污染的 GSM1k 数据集进行验证，实现模型推理能力的显著提升。

## 🚀 项目架构

项目采用了模块化设计，确保逻辑的高内聚与低耦合：

### 1. 核心训练模块
- **[train_grpo.py](train_grpo.py)**: 训练主入口。使用 `TRL` 库的 `GRPOTrainer` 进行强化学习。采用了 LoRA (Low-Rank Adaptation) 技术实现参数高效微调。
- **[reward_functions.py](reward_functions.py)**: 定义了多维度的奖励函数，包括准确性奖励、格式规范奖励以及推理步骤长度奖励，引导模型进行深入思考。

### 2. 数据与工具模块
- **[data_utils.py](data_utils.py)**: 数据大管家。负责数据集（GSM8K, GSM1k）的自动下载、本地持久化管理以及统一的格式化解析。
- **[evaluate.py](evaluate.py)**: 评估与可视化。支持双模型（训练前 vs 训练后）对比推理，结果自动保存为 JSON 并生成性能对比图表。

### 3. 环境与配置
- **[venv/](venv/)**: 预配置的 Python 虚拟环境，包含 PyTorch, Transformers, PEFT, TRL 等核心依赖。
- **`result/` & `result-score/`**: 自动生成的文件夹，用于存储推理日志和评分图表。

---

## 🛠️ 如何运行

### 1. 环境准备
确保你使用的是项目自带的虚拟环境：
```bash
source venv/bin/activate
```

### 2. 数据准备
你可以先手动运行数据脚本下载并缓存数据集：
```bash
python3 data_utils.py
```

### 3. 开始训练
运行 GRPO 训练脚本。训练完成后，模型权重将保存在 `qwen-grpo-gsm8k-final` 文件夹中：
```bash
python3 train_grpo.py
```

### 4. 性能评测
使用防污染的 GSM1k 数据集对比训练前后的效果。该脚本会生成 JSON 日志和对比柱状图：
```bash
python3 evaluate.py
```

---

## 📊 核心技术点

- **GRPO (Group Relative Policy Optimization)**: 相比传统 PPO，无需 Critic 模型，通过组内相对奖励优化，极大节省显存。
- **LoRA 微调**: 仅微调 Qwen 模型的 `q_proj`, `v_proj` 等核心线性层，保护预训练知识的同时实现快速对齐。
- **防污染验证**: 使用 **GSM1k** 数据集。这些题目在模型预训练阶段是不可见的，能够真实反映模型是否具备通用的数学推理能力，而非仅仅是“背题”。
