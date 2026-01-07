# 项目简历（PRACTICE 目录）

## 目录概览
- deepseek-vl_build_example1029/、GNNs/、Label Errors/、Recommender_System/、reinforcement_learning_bird/、test_api/
- 其他单文件/笔记本：数据筛选、推荐与图学习、强化学习、LLM 微调、语音 TTS/ASR 等实验脚本与 notebook。

## 多模态与大模型
- `deepseek-vl_build_example1029/`：FastAPI 部署 DeepSeek-VL2，多轮对话支持本地图片输入，含 `api_server.py`、`vl_utils.py` 推理封装。
- `GRPO_LLM_Fine_Tuning.ipynb`：基于 GRPO 的对齐/强化学习微调实验。
- `law_train.py`：使用 QLoRA 对 Qwen3-14B 进行法律咨询领域 SFT，含数据清洗、量化加载、LoRA 注入与 Trainer 训练脚本。
- `llm-emotion-helper_fine-tuning.ipynb`：情绪助手场景的指令微调实践。
- `Dataset Curation_multi_annotators.ipynb.ipynb`、`Entropy-Based Data Selection for LLMs.ipynb`、`Growing_Datasets.ipynb`：多标注者数据整理、基于熵的数据筛选与数据集扩充探索。
- `t5.ipynb`：T5 模型相关实验。

## 语音与多模态 API
- `qwen-asr-tts-realtime.py`：通义千问 TTS 流式语音播放示例，含音频播放链路。
- `test_api/test_asr_flash.py` 与 `welcome.mp3`：DashScope 多模态对话接口的 ASR 调用示例。

## 图学习与推荐系统
- `GNNs/`：Graph Neural Networks 实验（含数据、保存模型、可视化图像与多个训练 notebook）。
- `Recommender_System/`：推荐系统系列 notebook（item-item、冷启动、隐式反馈），附实验数据与展示 PPT。

## 强化学习
- `reinforcement_learning_bird/`：Double DQN/Deep Q-Learning 训练 Flappy Bird，含运行说明、测试脚本与模型实现。
- `强化学习.ipynb`：强化学习相关笔记与实验。

## NLP 与分类
- `金融数据分类基于（FinBERT).ipynb`：基于 FinBERT 的金融文本分类实验。
- `rnn_from_scratch.ipynb`、`transformer_from_scratch.py`：从零实现 RNN 与 Transformer，含详细调试输出来解析模型内部。
- `Graph Neural Networks.ipynb`（位于 `GNNs/`）与其他基础网络 notebook：图网络与深度学习基础实践。

## 数据质量与错误分析
- `Label Errors/`：学生成绩数据集的标签错误分析 notebook，含示例 CSV 数据。

## 其他
- 其他脚本：`part_*`（推荐系统分章节）、`test_bird.py` 等保持在各自目录，可按路径直接运行。
- `README.md` 与 `recommender_learning_outcomes.pptx`、`50039453_Jiayi Zheng.pptx` 等文件记录了项目备注与展示材料。
