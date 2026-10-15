#!/usr/bin/env python
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))

import whisper
import torch
import numpy as np
from collections import OrderedDict
from audio_duration_detector import AudioDurationDetector
from typing import Dict, Any, Tuple
from transformers import pipeline
import opencc
from homophone_corrector import HomophoneCorrector


def remove_duplicates(text: str) -> str:
    sentences = [s.strip() for s in text.split('《') if s.strip()]
    unique_sentences = list(OrderedDict.fromkeys(sentences))
    return '《' + '《'.join(unique_sentences) if unique_sentences else text


def clean_text(text: str) -> str:
    text = text.strip()
    text = ' '.join(text.split())
    return text


def clean_text_endings(text: str) -> str:
    """
    清理文本末尾的重复和乱码
    
    Args:
        text: 输入文本
        
    Returns:
        清理后的文本
    """
    import re
    
    if not text:
        return text
    
    # 1. 移除末尾的不完整句子
    incomplete_patterns = [
        r'需要我帮你把.*$',
        r'吹烟鸟鸡.*$',
        r'炊烟鸟雀.*$',
        r'不是萧瑟的它.*$',
        r'不是消色的它.*$',
    ]
    
    for pattern in incomplete_patterns:
        text = re.sub(pattern, '', text)
    
    # 2. 移除末尾的重复短语
    repeated_patterns = [
        r'(静静美好)\s+\1+$',
        r'(不慌不忙)\s+\1+$',
        r'(不急不徐)\s+\1+$',
        r'(在时光里)\s+\1+$',
    ]
    
    for pattern in repeated_patterns:
        text = re.sub(pattern, r'\1', text)
    
    # 3. 移除末尾的标点符号重复
    text = re.sub(r'[，。！？、；：]+\s*$', '。', text)
    
    # 4. 确保文本以句号结尾
    if text and not text[-1] in '。！？':
        text = text + '。'
    
    # 5. 移除末尾多余的空格
    text = text.strip()
    
    return text


def convert_to_simplified(text: str) -> str:
    """
    繁简转换，确保所有繁体字都转换为简体字
    
    Args:
        text: 输入文本
        
    Returns:
        简体中文文本
    """
    import re
    
    try:
        converter = opencc.OpenCC('t2s')
        converted_text = converter.convert(text)
        
        # 额外处理一些常见的繁简字问题
        extra_rules = {
            '著': '着',
            '带著': '带着',
            '挂著': '挂着',
            '咧著': '咧着',
            '点著': '踮着',
            '购不著': '摘不着',
        }
        
        for old, new in extra_rules.items():
            converted_text = converted_text.replace(old, new)
        
        # 处理连续的"著"字
        converted_text = re.sub(r'著+', '着', converted_text)
        
        return converted_text
    except Exception as e:
        print(f"繁简转换警告: {str(e)}")
        return text


def post_process_punctuation(text: str) -> str:
    """
    标点符号后处理，优化句号和逗号的使用
    
    Args:
        text: 识别结果文本
        
    Returns:
        优化后的文本
    """
    import re
    
    text = text.replace(',', '，')
    text = text.replace('.', '。')
    text = text.replace('!', '！')
    text = text.replace('?', '？')
    text = text.replace(':', '：')
    text = text.replace(';', '；')
    
    text = re.sub(r'，{2,}', '，', text)
    
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+([，。！？：；])', r'\1', text)
    
    return text

def print_section(title: str, content: str = None):
    print(f"\n{'='*60}")
    print(title)
    print(f"{'='*60}")
    if content:
        print(content)


def select_whisper_model(device: str, total_memory: float, duration: float = None) -> str:
    """
    根据设备、内存和音频时长选择合适的Whisper模型
    
    Args:
        device: 设备类型（cuda或cpu）
        total_memory: GPU内存大小（GB）
        duration: 音频时长（秒）
    
    Returns:
        模型名称
    """
    if device != "cuda":
        if duration and duration > 600:
            return "openai/whisper-small"
        return "openai/whisper-medium"
    
    if total_memory < 4.0:
        return "openai/whisper-tiny"
    elif total_memory < 6.0:
        if duration and duration > 600:
            return "openai/whisper-medium"
        return "openai/whisper-large-v3"
    elif total_memory < 8.0:
        if duration and duration > 600:
            return "openai/whisper-medium"
        return "openai/whisper-large-v3"
    else:
        if duration and duration > 600:
            return "openai/whisper-medium"
        return "openai/whisper-large-v3"


def load_audio_file(audio_file: str) -> Tuple[np.ndarray, float]:
    import os
    normalized_path = os.path.normpath(audio_file)
    audio = whisper.load_audio(normalized_path, sr=16000)
    audio = audio.astype(np.float32)
    duration = len(audio) / 16000.0
    return audio, duration


def process_transcription_result(result: Dict[str, Any]) -> str:
    homophone_corrector = HomophoneCorrector()
    
    if 'chunks' not in result:
        chinese_text = result['text']
        chinese_text = clean_text(chinese_text)
        chinese_text = convert_to_simplified(chinese_text)
        chinese_text = homophone_corrector.correct(chinese_text)
        chinese_text = post_process_punctuation(chinese_text)
        chinese_text = clean_text_endings(chinese_text)
        print_section("识别的中文内容：", chinese_text)
        return chinese_text
    
    chunks = result['chunks']
    print_section("识别的中文内容（分段）：")
    
    unique_segments = []
    seen_texts = set()
    
    for chunk in chunks:
        if 'text' not in chunk or 'timestamp' not in chunk:
            continue
        
        text = clean_text(chunk['text'])
        if not text or text in seen_texts:
            continue
        
        seen_texts.add(text)
        start, end = chunk['timestamp']
        simplified_text = convert_to_simplified(text)
        simplified_text = homophone_corrector.correct(simplified_text)
        print(f"[{start:.1f}s - {end:.1f}s] {simplified_text}")
        unique_segments.append(text)
    
    if not unique_segments:
        return ""
    
    chinese_text = ' '.join(unique_segments)
    chinese_text = clean_text(chinese_text)
    chinese_text = convert_to_simplified(chinese_text)
    chinese_text = homophone_corrector.correct(chinese_text)
    chinese_text = post_process_punctuation(chinese_text)
    chinese_text = clean_text_endings(chinese_text)
    
    print_section("去重后的中文文本：", chinese_text)
    return chinese_text


def save_to_file(audio_file: str, chinese_text: str, result: Dict[str, Any], duration: float, processing_time: float):
    """
    将识别结果保存到UTF-8编码的文本文件
    
    Args:
        audio_file: 音频文件路径
        chinese_text: 识别的中文文本
        result: Whisper识别结果（包含时间戳）
        duration: 音频时长
        processing_time: 处理时间
    """
    import datetime
    
    # 生成输出文件名
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"recognition_result_{timestamp}.txt"
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("普通话语音识别结果\n")
            f.write("="*80 + "\n\n")
            
            # 文件信息
            f.write("【文件信息】\n")
            f.write(f"音频文件: {audio_file}\n")
            f.write(f"音频时长: {duration:.2f} 秒\n")
            f.write(f"处理时间: {processing_time:.2f} 秒\n")
            f.write(f"识别时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"编码格式: UTF-8\n\n")
            
            # 分段识别结果（带时间戳）
            if 'chunks' in result:
                f.write("【分段识别结果（带时间戳）】\n")
                f.write("-"*80 + "\n")
                for i, chunk in enumerate(result['chunks'], 1):
                    if 'text' in chunk and 'timestamp' in chunk:
                        start, end = chunk['timestamp']
                        text = convert_to_simplified(clean_text(chunk['text']))
                        f.write(f"[{i:3d}] [{start:6.1f}s - {end:6.1f}s] {text}\n")
                f.write("\n")
            
            # 完整识别文本
            f.write("【完整识别文本】\n")
            f.write("-"*80 + "\n")
            f.write(chinese_text + "\n\n")
            
            # 统计信息
            f.write("【统计信息】\n")
            f.write("-"*80 + "\n")
            char_count = len(chinese_text)
            word_count = len(chinese_text.split())
            f.write(f"字符数: {char_count}\n")
            f.write(f"词数: {word_count}\n")
            f.write(f"平均识别速度: {duration/processing_time:.2f}x 实时速度\n")
            f.write("\n")
            
            f.write("="*80 + "\n")
            f.write("识别完成\n")
            f.write("="*80 + "\n")
        
        print(f"\n识别结果已保存到: {output_file}")
        return output_file
    except Exception as e:
        print(f"\n保存文件时出错: {str(e)}")
        return None


def main():
    print("="*60)
    print("普通话语音识别工具 - 高质量语音转文本")
    print("="*60)
    print("\n【系统配置】")
    print(f"支持音频格式: WAV, MP3, M4A, FLAC, OGG, WEBM")
    print(f"支持时长范围: 1秒 - 3600秒（1小时）")
    print(f"输出编码格式: UTF-8")
    print(f"噪声抑制: 自动启用")
    print(f"时间戳信息: 完整支持")
    print(f"内存优化: 自动启用（超长音频）")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n使用设备: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU 内存: {total_memory:.2f} GB")
        
        torch.cuda.empty_cache()
        print("已清理CUDA缓存")
    else:
        print("CPU模式")
    
    detector = AudioDurationDetector("audio_duration_config.json")
    audio_file = '21.wav'
    
    print(f"\n音频文件路径: {audio_file}")
    
    analysis_result = detector.analyze_audio_file(audio_file)
    detector.log_processing_info(audio_file, analysis_result)
    
    if not analysis_result['file_exists']:
        print(f"错误: 音频文件不存在！")
        return
    
    if analysis_result['error']:
        print(f"警告: {analysis_result['error']}")
        print("将使用默认处理策略继续...")
    
    strategy = analysis_result.get('strategy', {})
    strategy_params = detector.get_strategy_parameters(strategy)
    
    print(f"\n正在加载音频文件...")
    try:
        audio, duration = load_audio_file(audio_file)
        print(f"音频加载完成！采样率: 16000 Hz, 时长: {duration:.2f} 秒")
        
        if duration > 600:
            print(f"⚠️ 检测到超长音频（{duration:.2f}秒），将启用内存优化模式")
    except Exception as e:
        print(f"错误加载音频: {str(e)}")
        return
    print(f"\n正在加载 Whisper 模型...")
    
    whisper_model = select_whisper_model(device, total_memory if torch.cuda.is_available() else 0, duration)
    model_size = "Tiny" if "tiny" in whisper_model else "Base" if "base" in whisper_model else "Small" if "small" in whisper_model else "Medium" if "medium" in whisper_model else "Large V3"
    print(f"使用Whisper {model_size}模型")
    
    if device == "cuda":
        print("启用GPU内存优化...")
        torch.cuda.empty_cache()
    
    if "large-v3" in whisper_model and device == "cuda":
        print("Large V3 模型内存优化模式已启用")
        torch_dtype = torch.float16
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "low_cpu_mem_usage": True,
            "use_safetensors": True
        }
    else:
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        model_kwargs = {
            "torch_dtype": torch_dtype
        }
    
    whisper_pipe = pipeline(
        "automatic-speech-recognition", 
        model=whisper_model, 
        device=device,
        model_kwargs=model_kwargs,
        return_timestamps=strategy_params.get("return_timestamps", True)
    )
    print("Whisper 模型加载完成！")
    
    print(f"\n正在识别中文内容...")
    print(f"使用处理策略: {strategy.get('name', 'default')}")
    
    if duration > 600:
        print("⚠️ 超长音频处理预计需要较长时间，请耐心等待...")
        print(f"预计处理时间: {duration / 10:.0f} - {duration / 5:.0f} 秒")
    else:
        print("(这可能需要几分钟时间，请耐心等待...)")
    
    import time
    start_time = time.time()
    
    generation_kwargs = {
        "language": "chinese", 
        "task": "transcribe",
        "temperature": 0.0,
        "no_repeat_ngram_size": 5,
        "return_timestamps": True
    }
    
    whisper_params = {}
    if strategy_params.get("chunk_length_s") is not None:
        whisper_params["chunk_length_s"] = strategy_params["chunk_length_s"]
    if strategy_params.get("stride_length_s") is not None:
        whisper_params["stride_length_s"] = strategy_params["stride_length_s"]
    
    if duration > 600 and device == "cuda":
        print("启用GPU内存优化...")
        torch.cuda.empty_cache()
    
    result = whisper_pipe(audio, generate_kwargs=generation_kwargs, **whisper_params)
    
    if device == "cuda":
        torch.cuda.empty_cache()
    
    end_time = time.time()
    print(f"\n语音识别完成！用时: {end_time - start_time:.2f} 秒")
    print(f"处理速度: {duration / (end_time - start_time):.2f}x 实时速度")
    
    chinese_text = process_transcription_result(result)
    
    print_section("中文识别结果：", chinese_text)
    
    output_file = save_to_file(audio_file, chinese_text, result, duration, end_time - start_time)
    
    print("\n识别完成！")
    
    print("\n" + "="*60)
    print("自动运行准确率分析...")
    print("="*60)
    
    try:
        from accuracy_analyzer import AccuracyAnalyzer
        
        analyzer = AccuracyAnalyzer()
        
        default_ref_file = os.environ.get(
            "ACCURACY_REF_FILE", r"correct_text.txt"
        )
        print(f"\n加载参考文本: {default_ref_file}")
        
        if analyzer.load_reference_text(default_ref_file):
            if output_file:
                print(f"加载识别结果: {output_file}")
                if analyzer.load_recognition_result_file(output_file):
                    report = analyzer.generate_report()
                    print("\n" + report)
                    
                    report_file = "accuracy_analysis_report.txt"
                    analyzer.save_report(report_file)
                    print("\n准确率分析完成！")
                else:
                    print("警告: 加载识别结果文件失败，跳过准确率分析")
            else:
                print("警告: 未生成识别结果文件，跳过准确率分析")
        else:
            print("警告: 加载参考文本失败，跳过准确率分析")
    except Exception as e:
        print(f"\n准确率分析过程中出现错误: {str(e)}")
        print("跳过准确率分析")


if __name__ == "__main__":
    main()
