"""
Fine-tuning of llm in the field of legal consultation
"""
#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install -q accelerate==1.11.0 peft==0.17.1 bitsandbytes==0.48.2 transformers==4.57.1 trl==0.24.0')
get_ipython().system('pip install evaluate==0.4.6')


# In[1]:


import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

# ========================
# 环境与路径
# ========================
MODEL_PATH = "./Qwen3-14B"   # 你已下载的模型目录
DATA_PATH  = "./dataset_sft" # 你保存的HF Dataset目录（见下方第4步的保存）
OUTPUT_DIR = "./qlora-qwen3-14b"

# ========================
# 4-bit 量化配置（QLoRA）
# ========================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,  # 若不支持bf16，可改为 torch.float16
)

# ========================
# 加载分词器与模型
# ========================
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    use_fast=False,               # Qwen 系列通常建议关闭 fast
    trust_remote_code=True
)
# 确保有 EOS
if tokenizer.eos_token is None:
    tokenizer.eos_token = "<|endoftext|>"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# 让模型进入 k-bit 训练安全模式（冻结 norm 等）
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

# ========================
# LoRA 配置（常用的一组 target_modules）
# ========================
peft_config = LoraConfig(
    r=64,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
)

model = get_peft_model(model, peft_config)


# In[2]:


import json
from datasets import Dataset

data = []
with open("./law_corpus.json", "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line.strip())
        question = item.get("question", "").strip()
        answer = " ".join(item.get("answers", [])).strip()

        # ✅ 过滤：跳过过短的问答（字数 < 20）
        if len(question) < 20 or len(answer) < 20:
            continue

        # ✅ 构造 Qwen3 格式问答文本
        text = (
            "<|im_start|>system\n"
            "你是一个法律顾问助手。<|im_end|>\n"
            f"<|im_start|>user\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n{answer}<|im_end|>\n"
        )
        data.append({"text": text})

# 转为 Hugging Face Dataset 格式
dataset = Dataset.from_list(data)

# 划分训练 / 验证集
dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_data = dataset["train"]
test_data  = dataset["test"]

print(f"共加载 {len(train_data)} 条训练数据，{len(test_data)} 条测试数据。")
print(train_data[0]["text"])

# ✅ 保存为 Hugging Face Dataset 格式，供 train.py 使用
train_data.save_to_disk("./dataset_sft/train")
test_data.save_to_disk("./dataset_sft/test")

print("数据集已保存到 ./dataset_sft/")


# In[ ]:


# ========================
# 数据加载
# 你前面生成了 train/test 的 Dataset，这里假设你已保存到 DATA_PATH
# （若你当前变量还在内存里，请先用 train_data.save_to_disk(DATA_PATH + "/train") 等方式保存）
# ========================
train_dataset = load_from_disk(os.path.join(DATA_PATH, "train"))
eval_dataset  = load_from_disk(os.path.join(DATA_PATH, "test"))

# 将 text -> input_ids（pack 到 max_length，按需截断）
def tokenize_fn(examples):
    texts = examples["text"]
    # Qwen3 已经是带模板的纯SFT，直接当作连续文本因果建模
    toks = tokenizer(
        texts,
        max_length=4096,        # 视显存与场景调整（2k/4k/8k）
        truncation=True,
        padding=False,
        return_attention_mask=True
    )
    return toks

train_dataset = train_dataset.map(tokenize_fn, batched=True, remove_columns=train_dataset.column_names)
eval_dataset  = eval_dataset.map(tokenize_fn,  batched=True, remove_columns=eval_dataset.column_names)

train_dataset = train_dataset.select(range(min(2000, len(train_dataset))))
eval_dataset  = eval_dataset.select(range(min(500, len(eval_dataset))))

# LM数据整理器（不会动态mask）
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ========================
# 训练参数
# ========================
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,       # QLoRA 常用小batch，靠梯度累积
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,                  # QLoRA 常见 2e-4 ~ 3e-4
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=10,
    save_steps=500,
    evaluation_strategy="steps",
    eval_steps=500,
    save_total_limit=3,
    bf16=True,                           # 若不支持bf16，改为 fp16=True
    fp16=False,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",            # 结合 bitsandbytes 的优化器
    dataloader_num_workers=4,
    report_to="none",                    # 可改 "wandb"
)

# ========================
# 训练器
# ========================
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# 打印可训练参数（验证 QLoRA 是否生效）
def print_trainable_params(model):
    trainable, total = 0, 0
    for _, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    print(f"Trainable params: {trainable/1e6:.2f}M | Total params: {total/1e6:.2f}M | Ratio: {100*trainable/total:.4f}%")

print_trainable_params(model)

# 开训
trainer.train()

# 只保存 LoRA 适配器（最省空间）
trainer.model.save_pretrained(os.path.join(OUTPUT_DIR, "lora"))
tokenizer.save_pretrained(OUTPUT_DIR)

print("Done. LoRA adapters saved to:", os.path.join(OUTPUT_DIR, "lora"))


# In[ ]:




