# 扬州方言中医智能问诊 ASR 纠错系统

面向中医门诊的方言语音识别与医学术语纠错系统：患者使用扬州方言陈述症状，ASR 识别出的同音/近音错字（如「当归」→「荡鬼」、「气虚」→「七区」）由 **N-best Closest 约束解码** 模块在医学术语词典与本地大模型的辅助下完成纠正，且**最终输出严格限定在 ASR 的物理候选路径内**，从机制上杜绝大模型幻觉。

## 整体架构

```
医患对话音频
   │
   ▼
ASR 语音识别（Whisper large-v3 / faster-whisper / FunASR SenseVoiceSmall）
   │  N-best 候选列表 + 置信度
   ▼
┌─────────────────────────────────────────────┐
│  N-best Closest 约束纠错（correction/）        │
│  1. 字典预热：扬州方言/中医术语词典加权          │
│  2. LLM 自由推理：本地部署 qwen2.5 生成纠错参考  │
│  3. 最近距离映射：编辑距离把语义意图吸附回        │
│     ASR 物理候选路径，输出得分最高的原始候选      │
└─────────────────────────────────────────────┘
   │
   ▼
结构化问诊文本 → 本地大模型信息抽取 → 问诊报告
```

核心打分公式（`correction/utils.py`）：

```
Final_Score = ASR_Conf × w1 + Dict_Bonus × w2 − Levenshtein_Dist × w3
```

## 目录结构

```
├── asr/                     # 语音识别
│   ├── train_whisper_simple.py   # Whisper large-v3 增量训练（LoRA-free 简化版）
│   └── main_translate.py         # 按显存/时长自动选型的转写主流程
├── correction/              # 扬州方言 ASR 纠错模块（核心）
│   ├── error_correction.py       # N-best Closest 纠错流水线
│   ├── utils.py                  # 编辑距离 / 词典匹配 / 打分
│   ├── llm_client.py             # 本地 LLM（Ollama）客户端，含降级策略
│   ├── test_pipeline.py          # 含扬州话典型错词场景的单元测试
│   ├── plan.md / deployment_plan.md
│   ├── prepare_offline_model.bat # 拉取 qwen2.5:1.5b 并打包离线模型
│   └── hospital_install.bat      # 医院断网终端一键部署（Ollama + 模型 + 自启）
├── demo/
│   └── app_gradio.py             # Gradio 演示：SenseVoiceSmall + fsmn-vad + LLM 总结
└── docs/                    # 设计文档（含完整 GUI/服务端架构设计）
```

## 快速开始

```bash
pip install -r requirements.txt

# 运行纠错模块单元测试（Mock LLM，无需本地大模型即可验证核心算法）
cd correction
python test_pipeline.py
```

启动 Gradio 演示（需要设置 `DEEPSEEK_API_KEY` 环境变量，或自行替换 LLM 接入点）：

```bash
python demo/app_gradio.py
```

Whisper 训练脚本需要自备方言语音数据集与对应 metadata JSON：

```bash
python asr/train_whisper_simple.py --help
```

## 医院离线部署

`correction/hospital_install.bat` 面向完全断网的医院终端：

1. 准备 `OllamaSetup.exe`（官网下载，因体积原因未随仓库分发）
2. 运行 `correction/prepare_offline_model.bat` 在联网机上拉取 `qwen2.5:1.5b` 并导出 `ollama_model_offline\`
3. 将脚本 + 安装包 + 模型目录放入同一目录，双击 `hospital_install.bat` 即可完成静默安装、模型导入、开机自启（对医生无感）

LLM 不可用时 `llm_client.py` 会自动降级为纯词典 + ASR 置信度打分模式，纠错仍可部分完成。

## 已知缺失

以下内容因原始工作目录未归档而缺失，完整设计见 `docs/`：

- 前端 GUI `speech_to_text_ui_backup.py` 与后端引擎 `speech_to_text_plugin.py`（PyQt5 实时转写界面，`docs/实施方案.md` 与 `docs/问诊系统框架流程图.md` 含逐类逐方法设计）
- `asr/main_translate.py` 引用的辅助模块：`audio_duration_detector`、`homophone_corrector`、`accuracy_analyzer`（该脚本暂不可独立运行）

## 语料说明

扬州方言语料（中国语言资源保护工程录音素材）涉及版权与使用授权，未随本仓库分发。方言词表样例见 `docs/中医智能问诊系统整体框架设计文档2.pdf` 第 3.4 节。

## License

MIT
