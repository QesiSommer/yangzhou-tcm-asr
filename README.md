# 扬州方言中医智能问诊 ASR 纠错系统

## 1. 项目解决什么问题

中医门诊中，患者常使用扬州方言陈述症状，通用 ASR 模型识别方言时会产生大量同音/近音错字——「当归」被识别成「荡鬼」、「气虚」被识别成「七区」。这些错字直接进入问诊记录会造成误诊风险。

直接用大模型改写原文也不可行：LLM 自由生成会「幻觉」扩写病情、更改原意，这在医疗场景中绝对禁止。

本项目的方案是 **N-best Closest 约束纠错**：结合扬州方言/中医术语词典与本地大模型的语义参考，把纠错意图吸附回 ASR 的 N-best 物理候选路径，**最终输出严格来自 ASR 原始候选**，从机制上保证零幻觉。

## 2. 主要功能

| 功能 | 位置 | 说明 |
|---|---|---|
| N-best Closest 纠错流水线 | `correction/error_correction.py` | 词典加权 → 本地 LLM 纠错参考 → 编辑距离映射 → 输出得分最高的 ASR 原始候选 |
| 综合打分引擎 | `correction/utils.py` | `Final_Score = ASR_Conf×w1 + Dict_Bonus×w2 − Levenshtein×w3`，权重可调 |
| 本地 LLM 客户端 | `correction/llm_client.py` | 对接 Ollama（qwen2.5:1.5b），LLM 不可用时自动降级为纯词典+置信度模式 |
| 单元测试 | `correction/test_pipeline.py` | 覆盖扬州话典型错词场景（当归→荡鬼、气虚→七区），Mock LLM 无需真模型 |
| Whisper 增量训练 | `asr/train_whisper_simple.py` | Whisper Large v3 简化训练脚本，内置示例元数据生成 |
| 语音转写主流程 | `asr/main_translate.py` | 按显存/时长自动选型 tiny→large-v3（依赖的辅助模块未归档，暂不可独立运行，见「已知缺失」） |
| 医院离线部署 | `correction/hospital_install.bat` | 断网终端一键完成 Ollama 静默安装、模型导入、开机自启（医生无感） |

## 3. 安装方法

环境要求：Python 3.10+；GPU 可选（演示与纠错在 CPU 上即可运行）。

```bash
git clone https://github.com/QesiSommer/yangzhou-tcm-asr.git
cd yangzhou-tcm-asr
pip install -r requirements.txt
```

可选组件：

- 本地大模型：安装 [Ollama](https://ollama.com) 后执行 `ollama pull qwen2.5:1.5b`（不装也能跑，`llm_client.py` 会自动降级）
- Whisper 训练：首次运行会通过 hf-mirror.com 自动下载模型权重（缓存目录由 `HF_HOME` 环境变量控制，默认 `~/.cache/huggingface`）

## 4. 使用方法

**运行纠错模块单元测试**（Mock LLM，无需本地大模型）：

```bash
cd correction
python test_pipeline.py
```

**体验纠错流水线演示**（推荐先跑这个，内置扬州话错词数据）：

```bash
cd correction
python error_correction.py
```

**Whisper 增量训练**（无命令行参数，运行后自动生成示例元数据到 `data\` 并开始 15 轮训练；实际训练请替换 `data/train_metadata.json` 为自己的方言数据）：

```bash
python asr/train_whisper_simple.py
```

**医院离线部署**：参考 `correction/deployment_plan.md`，在联网机运行 `correction/prepare_offline_model.bat` 导出离线模型包，然后按 `hospital_install.bat` 顶部注释操作。

> `asr/main_translate.py` 依赖的 `audio_duration_detector`、`homophone_corrector`、`accuracy_analyzer` 模块未随仓库归档，暂不可独立运行，完整设计见 `docs/`。

## 5. 输入输出示例

**输入**（ASR 返回的 N-best 候选列表 + 问诊上下文）：

```python
n_best_list = [
    {"text": "需要荡鬼调理", "conf": 0.80},   # ASR 最置信，但「荡鬼」是「当归」的方言同音错字
    {"text": "需要当归调理", "conf": 0.15},
    {"text": "需要当鬼调理", "conf": 0.05},
]
context = "患者自述：最近头晕乏力，面色萎黄，经量少色淡。"
```

**输出**（`python correction/error_correction.py` 实际运行结果）：

```
======================================================================
  TCM ASR 纠错报告
======================================================================
  LLM 自由纠错文本   : 需要当归调理
  最终输出           : 需要当归调理
  最终得分           : 0.2600
  权重配置           : w1=0.4, w2=0.2, w3=0.4
----------------------------------------------------------------------
  候选句                ASR   词典   编辑距    最终分
----------------------------------------------------------------------
  需要当归调理          0.150  [Y]      0   0.2600
  需要当鬼调理          0.050  [N]      1  -0.3800
  需要荡鬼调理          0.800  [N]      2  -0.4800
======================================================================
```

尽管「需要荡鬼调理」的 ASR 置信度最高（0.80），但它未命中中医术语词典、且与 LLM 纠错参考的编辑距离最大，综合得分最低——系统最终输出物理候选中的「需要当归调理」，而不是 LLM 自由生成的文本。

## 目录结构

```
├── asr/                     # 语音识别（训练 + 转写）
├── correction/              # 扬州方言 ASR 纠错模块（核心）
└── docs/                    # 设计文档（含完整 GUI/服务端架构设计）
```

## 已知缺失

以下内容因原始工作目录未归档而缺失，完整设计见 `docs/`：

- 前端 GUI `speech_to_text_ui_backup.py` 与后端引擎 `speech_to_text_plugin.py`（PyQt5 实时转写界面，`docs/实施方案.md` 与 `docs/问诊系统框架流程图.md` 含逐类逐方法设计）
- `asr/main_translate.py` 引用的辅助模块（见第 4 节说明）

## 语料说明

扬州方言语料（中国语言资源保护工程录音素材）涉及版权与使用授权，未随本仓库分发。方言词表样例见 `docs/中医智能问诊系统整体框架设计文档2.pdf` 第 3.4 节。

## License

MIT
