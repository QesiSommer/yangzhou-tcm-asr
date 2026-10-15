import os
import gradio as gr
from funasr import AutoModel
import re
import numpy as np
from openai import OpenAI
import time

# ================= 1. 核心引擎初始化 =================
print(">>> 正在启动 ASR 听觉引擎...")
# 底层自带 fsmn-vad 防爆破护盾，防止长视频 OOM
model = AutoModel(
    model="iic/SenseVoiceSmall", 
    trust_remote_code=True, 
    vad_model="fsmn-vad", 
    vad_kwargs={"max_single_segment_time": 30000},
    device="cpu"
)

print(">>> 正在初始化 DeepSeek-V4-Pro 大脑引擎...")
# ⚠️ API KEY
LLM_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

llm_client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

# ================= 2. 基础工具与提示词函数 =================

def clean_sensevoice_tags(raw_text):
    """文本清洗核心：剔除控制标签"""
    pattern = r'<\|.*?\|>'
    cleaned_text = re.sub(pattern, '', raw_text)
    return cleaned_text.replace('  ', ' ').strip()

def extract_partial_summary(text):
    """功能 1：阶段性总结"""
    system_prompt = """
    你是一个专业的中医助手。请对以下这段问诊对话片段进行简要总结。
    
    输出要求：
    1. 总结这一阶段的对话大主题。
    2. 严格遵循格式：**[阶段大主题]**：[具体的总结内容]
    3. 语言要精炼，突出医疗关键信息，切忌编造上下文中没有的内容。
    """
    try:
        response = llm_client.chat.completions.create(
            model="deepseek-v4-pro",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"请简要总结以下对话片段：\n{text}"}
            ],
            extra_body={"enable_thinking": True},
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ 阶段总结失败: {str(e)}"

def extract_final_medical_record(text):
    """功能 2：全局病历"""
    system_prompt = """
    你是一个经验丰富的中医临床助理。你的任务是从一段可能带有方言口音、错别字或日常闲聊的语音识别文本中，提炼出结构化的【中医电子病历与医嘱】。
    
    请严格按照以下 Markdown 格式输出。如果未提到某项信息，请填写“未提及”。
    如果整段文本与医疗问诊毫无关系，请直接输出：“⚠️ 检测到当前对话非医疗问诊内容，无法生成病历。”
    
    ### 📝 智能病历摘要
    * **主诉症状：** (提取患者不舒服的地方)
    * **医生诊断：** (提取病名或病因分析)
    * **处方与用药：** (提取药名及用量)
    * **医嘱与禁忌：** (提取注意事项)
    """
    try:
        response = llm_client.chat.completions.create(
            model="deepseek-v4-pro",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"请提取以下问诊记录：\n{text}"}
            ],
            extra_body={"enable_thinking": True},
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ 全局病历生成失败: {str(e)}"

# ================= 3. 业务流 1：实时麦克风门诊 =================

def process_mic_chunk(audio_stream, state):
    """麦克风流式处理：严格执行 60 秒截断识别，不加任何额外干预逻辑"""
    if audio_stream is None:
        return state, "", ""

    sr, data = audio_stream
    # 倒入蓄水池
    state['buffer'] = np.concatenate([state['buffer'], data]) if state['buffer'].size else data
    
    current_seconds = len(state['buffer']) / sr

    # 严格按照你要求的 60 秒时间节点截取
    if current_seconds >= 60.0:
        print(f">>> 触发阶段节点 (60s)，开始分析...")
        res = model.generate(input=state['buffer'], language="auto", use_itn=True)
        clean_text = clean_sensevoice_tags(res[0]['text'])
        
        if len(clean_text) > 5:
            summary = extract_partial_summary(clean_text)
            state['partial_summaries'].append(summary)
            state['history_texts'].append(clean_text)
            
        # 清空蓄水池
        state['buffer'] = np.array([], dtype=np.int16) 

    history_display = "\n\n".join(state['history_texts'])
    summary_display = "\n\n".join([f"📍 {s}" for s in state['partial_summaries']])
    return state, history_display, summary_display

def finalize_mic_report(state):
    """麦克风关闭：生成全局病历"""
    if not state['history_texts']:
        return "⚠️ 未检测到有效内容，无法生成病历。"
    print(">>> 正在生成麦克风全局病历...")
    full_content = "\n".join(state['history_texts'])
    return extract_final_medical_record(full_content)

# ================= 4. 业务流 2：录像复盘分析 =================

def process_file(audio_file, video_file):
    """对上传的音视频文件进行【模拟阶段分析 + 全局分析】"""
    target_file = audio_file if audio_file else video_file
    if not target_file:
        yield "⚠️ 请上传音视频文件", "", ""
        return
    
    yield "⏳ 正在提取底层语音并进行识别...", "等待数据...", "等待数据..."
    
    # 1. 一次性获取全文件识别结果
    res = model.generate(input=target_file, language="auto", use_itn=True)
    full_clean_text = clean_sensevoice_tags(res[0]['text'])
    
    yield full_clean_text, "⏳ 语音转写完成，正在进行拆解与阶段性总结...", "等待数据..."
    
    # 2. 模拟“切段”：按句号拆分文本，大约每 100 字归为一阶段
    sentences = re.split(r'([。！？!?])', full_clean_text)
    chunks = []
    curr = ""
    for s in sentences:
        curr += s
        if len(curr) > 100:
            chunks.append(curr)
            curr = ""
    if curr: chunks.append(curr)

    partial_summaries = []
    # 3. 逐段调用大模型，实时动态刷新网页
    for i, chunk in enumerate(chunks):
        if len(chunk.strip()) < 5: continue
        print(f">>> 正在处理视频阶段 {i+1}/{len(chunks)} ...")
        summary = extract_partial_summary(chunk)
        partial_summaries.append(summary)
        
        # 实时用 yield 推送给网页
        current_sum_display = "\n\n".join([f"📍 {s}" for s in partial_summaries])
        yield full_clean_text, current_sum_display, f"⏳ 阶段总结 ({i+1}/{len(chunks)}) 完成，正在生成最终病历..."
        
    # 4. 生成一锤定音的全局病历
    print(">>> 正在生成视频全局病历...")
    final_report = extract_final_medical_record(full_clean_text)
    
    # 最后一次推流：展示完全体
    yield full_clean_text, current_sum_display, final_report

# ================= 5. 前端 UI 架构 (带有防盗门) =================

print(">>> 正在生成 V6.0 终极答辩界面...")
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    
    # ------------------ 防盗门（默认显示）------------------
    with gr.Column(visible=True) as login_pane:
        gr.Markdown("## 🔒 扬州话中医智能问诊系统 - 组会专属访问通道")
        gr.Markdown("为了保护 API 调用额度，请输入访问密码。")
        pwd_input = gr.Textbox(label="请输入授权密码", type="password", placeholder="此处输入密码，验证成功后进入系统")
        login_btn = gr.Button("验证并进入工作站", variant="primary")
        login_msg = gr.Markdown("") # 显示报错信息

    # ------------------ 核心工作站（默认隐藏）------------------
    with gr.Column(visible=False) as main_pane:
        gr.Markdown("# 🩺 扬州话中医智能问诊V6")
        gr.Markdown("内置 DeepSeek-V4-Pro 深度思考引擎 | 支持实时麦克风阶段截断提炼 & 离线录像流式解析。")
        
        with gr.Tabs():
            
            # ---------- 选项卡 1：实时问诊 ----------
            with gr.TabItem("🔴 现场实时门诊 (麦克风)"):
                mic_state = gr.State({ 'buffer': np.array([], dtype=np.int16), 'history_texts': [], 'partial_summaries': [] })
                
                with gr.Row():
                    with gr.Column(scale=1):
                        audio_input = gr.Audio(sources=["microphone"], type="numpy", streaming=True, label="1. 点击麦克风开始持续录音")
                        stop_btn = gr.Button("🛑 2. 结束就诊 (生成全局病历)", variant="primary")
                        
                    with gr.Column(scale=2):
                        with gr.Row():
                            mic_asr_out = gr.Textbox(label="🕒 ASR 阶段转写流 (每 60 秒截断)", lines=6)
                            mic_part_out = gr.Textbox(label="💡 AI 阶段性便签", lines=6)
                        mic_final_out = gr.Markdown(label="📜 最终全局导出：中医电子病历")

                audio_input.stream(
                    fn=process_mic_chunk, inputs=[audio_input, mic_state], outputs=[mic_state, mic_asr_out, mic_part_out]
                )
                stop_btn.click(
                    fn=finalize_mic_report, inputs=[mic_state], outputs=[mic_final_out]
                )

            # ---------- 选项卡 2：录像复盘 ----------
            with gr.TabItem("📁 录音/录像存档复盘"):
                with gr.Row():
                    with gr.Column(scale=1):
                        file_audio_input = gr.Audio(sources=["upload"], type="filepath", label="上传音频存档")
                        file_video_input = gr.Video(label="或上传问诊录像")
                        file_submit_btn = gr.Button("🚀 开始智能解析", variant="primary")
                        
                    with gr.Column(scale=2):
                        with gr.Row():
                            file_asr_out = gr.Textbox(label="🕒 完整问诊逐字稿", lines=6)
                            file_part_out = gr.Textbox(label="💡 阶段还原提炼", lines=6)
                        file_final_out = gr.Markdown(label="📜 全局结构化病历")

                file_submit_btn.click(
                    fn=process_file, 
                    inputs=[file_audio_input, file_video_input], 
                    outputs=[file_asr_out, file_part_out, file_final_out]
                )

    # ------------------ 密码验证逻辑 ------------------
    # 仅演示用门禁，非安全边界；可通过 DEMO_PASSWORD 环境变量修改
    demo_password = os.environ.get("DEMO_PASSWORD", "tiaozhanbei2026")

    def verify_password(pwd):
        if pwd == demo_password:
            # 密码正确：隐藏登录层，显示主界面，清空报错
            return gr.update(visible=False), gr.update(visible=True), ""
        else:
            # 密码错误：保持界面，显示红色报错
            return gr.update(visible=True), gr.update(visible=False), "❌ **暗号错误，请检查！**"

    login_btn.click(
        fn=verify_password, 
        inputs=pwd_input, 
        outputs=[login_pane, main_pane, login_msg]
    )

# ================= 6. 纯净启动 =================
# 坚决不使用 auth 参数，避免魔搭环境冲突
demo.launch(server_name="0.0.0.0", server_port=7860)