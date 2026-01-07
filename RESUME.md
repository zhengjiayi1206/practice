# 郑佳毅 — 项目型简历

## 核心能力
- 端到端 AI 工程：从数据清洗/筛选 → 训练/微调 → 推理服务化 → 实验复现与汇报材料沉淀。
- 大模型与多模态：参数高效微调（QLoRA/LoRA）、多模态对话（图像输入）、对齐方法探索（GRPO）。
- 算法落地与系统化实验：推荐系统（相似度/冷启动/隐式反馈）、图神经网络训练与可视化、强化学习（Double DQN）。
- 工程实现：FastAPI 服务、GPU/MPS/CUDA 加速推理与训练脚本化、接口调用与音频流处理。

## 代表项目（基于本目录产出）
- DeepSeek-VL2 多模态对话 API（`deepseek-vl_build_example1029/`）
  - 将视觉语言模型封装为可调用的 FastAPI 服务，支持多轮对话与本地图片路径输入；提供健康检查与输入校验，推理侧包含显存清理与资源回收逻辑。
  - 关键词：FastAPI、Pydantic、Transformers/DeepSeek-VL2、CUDA 推理、服务化与工程健壮性。

- Qwen3-14B 法律咨询领域微调（`law_train.py`）
  - 搭建 QLoRA 微调训练流程：4-bit 量化加载（BitsAndBytes）、LoRA 注入、SFT 数据构造与过滤、训练/验证划分、Trainer 训练与检查点保存。
  - 关键词：PyTorch、Transformers、PEFT/LoRA、BitsAndBytes（4-bit NF4）、梯度累积、评估与复现实验。

- LLM 对齐与微调实验（`GRPO_LLM_Fine_Tuning.ipynb`、`llm-emotion-helper_fine-tuning.ipynb`、`t5.ipynb`）
  - 进行对齐方法（GRPO）与指令微调任务探索，沉淀可复现的 notebook 实验链路与结果记录。
  - 关键词：对齐/奖励优化思路、SFT、实验设计与迭代。

- 推荐系统体系化实验（`Recommender_System/`）
  - 覆盖 Item-Item 协同过滤、冷启动问题处理、隐式反馈推荐三条主线；配套数据与结果展示材料，形成从原理到实验的完整闭环。
  - 关键词：相似度建模、冷启动策略、隐式反馈、可解释分析与展示。

- 图神经网络实践（`GNNs/`）
  - 进行 GNN 训练实验与可视化，包含数据准备、训练流程、模型保存与结果展示。
  - 关键词：图表示学习、训练管线、实验管理与可视化。

- 强化学习 Double DQN：Flappy Bird（`reinforcement_learning_bird/torch_DDQL.py`）
  - 基于 PyTorch 实现 Double DQN 训练主循环，包含经验回放、目标网络同步、ε-greedy 探索与图像预处理；支持 CUDA/MPS/CPU 设备自适应。
  - 关键词：Deep RL、Replay Buffer、Target Network、训练稳定性与工程实现。

- 语音与多模态接口链路（`qwen-asr-tts-realtime.py`、`test_api/test_asr_flash.py`）
  - 实现 TTS 流式音频播放与 ASR 调用示例，完成从接口请求到音频解码/播放的闭环验证。
  - 关键词：DashScope/Qwen、流式响应、音频处理、端到端联调。

- 数据质量与错误分析（`Label Errors/` + 相关 notebook）
  - 对结构化数据进行标签错误分析与质量排查，并在多标注者与数据筛选主题上做方法探索（熵筛选/数据集扩充等）。
  - 关键词：数据质量、错误分析、数据筛选与整理。

## 技术栈（在本目录中体现）
- 语言与框架：Python、PyTorch、FastAPI、Pydantic
- 大模型生态：Transformers、PEFT/LoRA、BitsAndBytes（8bit/4bit 量化）、Trainer/训练脚本化
- 方向覆盖：多模态对话、LLM 微调/对齐、推荐系统、GNN、强化学习、ASR/TTS、文本分类与数据质量
