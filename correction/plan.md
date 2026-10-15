\# 项目规划书：基于本地大模型与 N-best Closest 策略的中医 ASR 纠错模块



\## 一、 项目背景与应用场景 (Background \& Application)

本项目为“中医智能问诊系统”的核心后处理模块。

\- \*\*痛点\*\*：由于患者常使用扬州方言，前端 ASR（Faster Whisper）识别极易产生同音/近音字错误（如“当归”被识别为“荡鬼”）。

\- \*\*挑战\*\*：直接使用大模型（LLM）纠错会导致严重的“幻觉”（如随意扩写病情、更改原意），这在医疗场景中是绝对禁止的。

\- \*\*目标\*\*：实现“严格意义上的补齐与纠错”。系统需结合中医知识和上下文，将 ASR 的错误文本纠正，但\*\*最终输出必须严格限制在 ASR 的物理候选路径中\*\*，绝不允许模型自由编造。



\## 二、 核心架构与处理逻辑 (Core Architecture \& Logic)

本模块采用 \*\*N-best Closest 约束解码架构\*\*。整体处理流水线包含四个步骤：

1\. \*\*输入接收 (ASR N-best Input)\*\*：接收前端 ASR 传入的 N-best 候选列表及其初始置信度分数。

2\. \*\*字典预热 (Dictionary Weighting)\*\*：使用预设的扬州方言/中医术语词典对 N-best 列表进行快速扫描。若候选词中包含专业术语，则给予权重加分。

3\. \*\*大模型自由推理 (LLM Unconstrained Inference)\*\*：将最高分的候选词及历史上下文发送给本地部署的中医大模型（如神农/华佗），获取 LLM 自由生成的纠错文本。

4\. \*\*最近距离映射 (N-best Closest Mapping) \[核心]\*\*：将 LLM 生成的文本与 ASR 提供的 N-best 候选列表逐一对比。计算每项的“最终得分”，强制将大模型的语义意图吸附回物理发音路径上，输出得分最高的原始候选词。



\## 三、 核心算法与打分公式 (Algorithm)

我们需要实现一个多维度的综合打分函数 `calculate\_final\_score`。

针对 N-best 列表中的每一项候选句 $z$，打分公式如下：

`Final\_Score(z) = (ASR\_Conf \* W1) + (Dict\_Bonus \* W2) - (Levenshtein\_Dist(LLM\_text, z) \* W3)`



\- `ASR\_Conf`: 语音引擎给出的原始置信度概率（0\~1）。

\- `Dict\_Bonus`: 命中中医词典的加分项（如命中得 1.0，未命中 0）。

\- `Levenshtein\_Dist`: 编辑距离，即 LLM 自由生成的文本与候选句 $z$ 之间需要改动几个字符。距离越大，扣分越狠。

\- `W1, W2, W3`: 可调参的权重因子。



\## 四、 技术栈与依赖工具 (Tech Stack \& Tools)

\- \*\*语言\*\*：Python 3.10+

\- \*\*关键库\*\*：

&#x20; - `python-Levenshtein` 或 `editdistance`：用于极速计算编辑距离。

&#x20; - `openai` 库 或 `requests`：用于调用本地部署的、兼容 OpenAI API 格式的 vLLM 接口（大模型接口）。

\- \*\*LLM 环境\*\*：假设本地大模型运行在 `http://127.0.0.1:8000/v1`。



\## 五、 Claude Code 任务拆解与执行计划 (Execution Plan)

请 Claude 按照以下顺序逐步创建和完善文件：



\### Task 1: 创建底层工具库 `utils.py`

\- 引入编辑距离计算库。

\- 实现中医词典的 Mock 数据匹配逻辑（如 `check\_tcm\_dict(text)`）。

\- 实现核心的 `calculate\_final\_score(asr\_conf, dict\_bonus, edit\_dist, w1=0.4, w2=0.2, w3=0.4)` 函数。



\### Task 2: 创建大模型连接层 `llm\_client.py`

\- 封装一个调用本地大模型 API 的类 `LocalLLMClient`。

\- 实现 `generate\_correction(context, asr\_1best\_text)` 方法。要求提示词（Prompt）中明确告知大模型其角色为“中医语音纠错专家”，只需输出纠正后的句子即可，勿作解释。



\### Task 3: 创建核心流水线 `error\_correction.py`

\- 整合上述两个文件。

\- 实现主函数 `run\_correction\_pipeline(n\_best\_list, context)`。

\- 逻辑：接收形如 `\[{"text": "需要荡鬼", "conf": 0.8}, {"text": "需要当归", "conf": 0.2}]` 的输入 -> 查字典打分 -> 调 LLM -> 计算编辑距离与 Final\_Score -> 排序并返回最高分的字典项。



\### Task 4: 创建测试用例 `test\_pipeline.py`

\- 编写扬州话错词 Mock 数据（例如：ASR 听错“当归”为“荡鬼”、“气虚”为“七区”）。

\- 运行 pipeline，验证系统是否能成功根据中医 LLM 的输出，将结果 Closest 映射回正确的中医候选词上，并打印打分明细。



\---

请先阅读以上规划。如果理解无误，请直接开始执行 Task 1 和 Task 2。

