﻿#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Whisper Large v3 增量训练脚本
简化版本，避免 tiktoken_ext 依赖问题
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import re

# 设置环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))

# 临时修改导入路径，优先使用系统安装的库（仅本机调试用，路径不存在时自动跳过）
import sys
_LEGACY_SITE_PACKAGES = 'd:\\Python39\\lib\\site-packages'
if os.path.isdir(_LEGACY_SITE_PACKAGES):
    sys.path.insert(0, _LEGACY_SITE_PACKAGES)

def traditional_to_simplified(text):
    """
    将繁体中文转换为简体中文
    使用 opencc 库进行转换
    
    Args:
        text: 繁体中文文本
    
    Returns:
        简体中文文本
    """
    import opencc
    converter = opencc.OpenCC('t2s')  # 繁体转简体
    return converter.convert(text)

def calculate_wer(reference, hypothesis):
    """
    计算词错误率 (Word Error Rate)
    对于中文，使用 jieba 分词后计算
    
    Args:
        reference: 参考文本
        hypothesis: 预测文本
    
    Returns:
        WER 值（0-1 之间）
    """
    # 处理空字符串情况
    if not reference:
        return 1.0 if hypothesis else 0.0
    if not hypothesis:
        return 1.0
    
    # 尝试使用 jieba 分词（如果是中文）
    try:
        import jieba
        ref_words = list(jieba.cut(reference))
        hyp_words = list(jieba.cut(hypothesis))
    except ImportError:
        # 如果没有安装 jieba，使用空格分词
        ref_words = reference.split()
        hyp_words = hypothesis.split()
    
    # 计算编辑距离
    import numpy as np
    dp = np.zeros((len(ref_words) + 1, len(hyp_words) + 1))
    
    for i in range(len(ref_words) + 1):
        dp[i][0] = i
    for j in range(len(hyp_words) + 1):
        dp[0][j] = j
    
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j] + 1,    # 删除
                               dp[i][j-1] + 1,    # 插入
                               dp[i-1][j-1] + 1)  # 替换
    
    # 确保分母不为零
    max_len = max(len(ref_words), 1)
    wer = dp[len(ref_words)][len(hyp_words)] / max_len
    
    # 限制 WER 在合理范围内
    return min(wer, 1.0)

def calculate_cer(reference, hypothesis):
    """
    计算字符错误率 (Character Error Rate)
    
    Args:
        reference: 参考文本
        hypothesis: 预测文本
    
    Returns:
        CER 值（0-1之间）
    """
    # 处理空字符串情况
    if not reference:
        return 1.0 if hypothesis else 0.0
    if not hypothesis:
        return 1.0
    
    ref_chars = list(reference)
    hyp_chars = list(hypothesis)
    
    # 计算编辑距离
    import numpy as np
    dp = np.zeros((len(ref_chars) + 1, len(hyp_chars) + 1))
    
    for i in range(len(ref_chars) + 1):
        dp[i][0] = i
    for j in range(len(hyp_chars) + 1):
        dp[0][j] = j
    
    for i in range(1, len(ref_chars) + 1):
        for j in range(1, len(hyp_chars) + 1):
            if ref_chars[i-1] == hyp_chars[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j] + 1,    # 删除
                               dp[i][j-1] + 1,    # 插入
                               dp[i-1][j-1] + 1)  # 替换
    
    # 确保分母不为零
    max_len = max(len(ref_chars), 1)
    cer = dp[len(ref_chars)][len(hyp_chars)] / max_len
    
    # 限制 CER 在合理范围内
    return min(cer, 1.0)

# 尝试导入 whisper
try:
    import whisper
except ImportError as e:
    print(f"导入 whisper 失败: {e}")
    sys.exit(1)

class WhisperDataset(Dataset):
    """
    Whisper 数据集类
    """
    
    def __init__(self, metadata_path, max_mel_length=3000, max_text_length=448):
        """
        初始化数据集
        
        Args:
            metadata_path: 元数据文件路径
            max_mel_length: Mel 频谱图最大长度
            max_text_length: 文本最大长度
        """
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        # 获取分词器
        self.tokenizer = whisper.tokenizer.get_tokenizer(True)
        self.max_mel_length = max_mel_length
        self.max_text_length = max_text_length
        
        # 特殊 token
        self.sot_token = self.tokenizer.sot  # Start of transcript
        self.eot_token = self.tokenizer.eot  # End of transcript
        
        print(f"数据集初始化完成")
        print(f"样本数量: {len(self.metadata)}")
        print(f"Mel 最大长度: {self.max_mel_length}")
        print(f"文本最大长度: {self.max_text_length}")
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.metadata)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        
        Args:
            idx: 样本索引
        
        Returns:
            sample: 包含 mel、tokens、text 等的字典
        """
        # 获取元数据
        item = self.metadata[idx]
        audio_path = item['audio']
        text = item['text']
        
        # 加载音频
        audio = whisper.load_audio(audio_path)
        
        # 转换为 Mel 频谱图
        mel = whisper.log_mel_spectrogram(audio)
        
        # 处理 Mel 频谱图长度
        if mel.shape[-1] > self.max_mel_length:
            mel = mel[..., :self.max_mel_length]
        elif mel.shape[-1] < self.max_mel_length:
            padding = torch.zeros(mel.shape[0], self.max_mel_length - mel.shape[-1])
            mel = torch.cat([mel, padding], dim=-1)
        
        # 确保通道数正确 (80 channels)
        if mel.shape[0] != 80:
            print(f"警告: Mel 频谱图通道数不正确: {mel.shape[0]}")
            # 调整通道数
            if mel.shape[0] > 80:
                mel = mel[:80, :]
            else:
                padding = torch.zeros(80 - mel.shape[0], mel.shape[1])
                mel = torch.cat([mel, padding], dim=0)
        
        # Whisper 模型需要 128 通道输入
        # 添加额外的 48 通道作为填充
        if mel.shape[0] == 80:
            padding = torch.zeros(48, mel.shape[1])
            mel = torch.cat([mel, padding], dim=0)
        
        # 编码文本
        tokens = self.tokenizer.encode(text)
        tokens = [self.sot_token] + tokens + [self.eot_token]
        
        # 处理文本长度
        if len(tokens) > self.max_text_length:
            tokens = tokens[:self.max_text_length-1] + [self.eot_token]
        elif len(tokens) < self.max_text_length:
            tokens = tokens + [self.eot_token] * (self.max_text_length - len(tokens))
        
        return {
            'mel': mel,
            'tokens': torch.tensor(tokens, dtype=torch.long),
            'text': text,
            'audio_path': audio_path
        }

def create_data_loaders(train_metadata, val_metadata, batch_size=1):
    """
    创建数据加载器
    
    Args:
        train_metadata: 训练元数据路径
        val_metadata: 验证元数据路径
        batch_size: 批次大小
    
    Returns:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
    """
    print("创建数据加载器...")
    
    # 创建数据集
    train_dataset = WhisperDataset(train_metadata)
    val_dataset = WhisperDataset(val_metadata)
    
    # 创建数据加载器 - 优化内存使用
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # 使用单进程，减少内存开销
        pin_memory=False,  # 禁用 pin memory
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # 使用单进程，减少内存开销
        pin_memory=False,  # 禁用 pin memory
        drop_last=False
    )
    
    print(f"训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(val_dataset)} 样本")
    print(f"训练批次数: {len(train_loader)}")
    print(f"验证批次数: {len(val_loader)}")
    
    return train_loader, val_loader

def freeze_high_level(model):
    """
    冻结模型高层
    
    Args:
        model: Whisper 模型
    """
    print("冻结模型层...")
    
    # 冻结编码器所有层
    for i in range(len(model.encoder.blocks)):
        for param in model.encoder.blocks[i].parameters():
            param.requires_grad = False
    
    # 冻结解码器大部分层，只保留最后2层可训练
    for i in range(len(model.decoder.blocks) - 2):
        for param in model.decoder.blocks[i].parameters():
            param.requires_grad = False
    
    # 统计可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"总参数量: {total_params / 1e6:.2f}M")
    print(f"可训练参数量: {trainable_params / 1e6:.2f}M ({trainable_params/total_params*100:.1f}%)")

def train_epoch(model, train_loader, optimizer, criterion, device):
    """
    训练一个 epoch
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        criterion: 损失函数
        device: 设备
    
    Returns:
        avg_loss: 平均损失
    """
    model.train()
    total_loss = 0
    
    for batch in tqdm(train_loader, desc="训练"):
        # 移动数据到设备
        mel = batch['mel'].to(device, non_blocking=True)
        tokens = batch['tokens'].to(device, non_blocking=True)
        
        # 前向传播
        encoder_output = model.encoder(mel)
        
        # 解码器输入
        decoder_input = tokens[:, :-1]  # 移除最后一个 token
        
        # 解码器前向传播
        outputs = model.decoder(decoder_input, encoder_output)
        
        # 计算损失
        target = tokens[:, 1:]  # 目标是下一个 token
        
        loss = criterion(
            outputs.reshape(-1, outputs.size(-1)),
            target.reshape(-1)
        )
        
        # 反向传播
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 更新参数
        optimizer.step()
        
        total_loss += loss.item()
        
        # 释放内存
        del mel, tokens, encoder_output, decoder_input, outputs, target, loss
        if device == "cuda":
            torch.cuda.empty_cache()
    
    if len(train_loader) > 0:
        avg_loss = total_loss / len(train_loader)
    else:
        avg_loss = 0.0
    
    return avg_loss

def evaluate_epoch(model, val_loader, criterion, device):
    """
    验证一个 epoch
    
    Args:
        model: 模型
        val_loader: 验证数据加载器
        criterion: 损失函数
        device: 设备
    
    Returns:
        avg_loss: 平均损失
        val_wer: 词错误率
        val_cer: 字符错误率
    """
    model.eval()
    total_loss = 0
    all_references = []
    all_hypotheses = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="验证"):
            # 移动数据到设备
            mel = batch['mel'].to(device, non_blocking=True)
            tokens = batch['tokens'].to(device, non_blocking=True)
            texts = batch['text']
            
            # 前向传播（只使用编码器和解码器的基本前向传播）
            encoder_output = model.encoder(mel)
            decoder_input = tokens[:, :-1]
            outputs = model.decoder(decoder_input, encoder_output)
            
            # 计算损失
            target = tokens[:, 1:]
            loss = criterion(
                outputs.reshape(-1, outputs.size(-1)),
                target.reshape(-1)
            )
            total_loss += loss.item()
            
            # 生成预测（使用贪婪解码 - 与训练方式一致）
            for i in range(mel.size(0)):
                # 提取单个样本
                mel_single = mel[i:i+1]
                
                # 获取参考文本
                reference = texts[i]
                
                # 使用 Whisper 的 decode 方法，强制使用中文
                try:
                    # 使用模型的 decode 方法，指定语言和任务
                    result = model.decode(mel_single, 
                                         language="zh",  # 强制使用中文
                                         task="transcribe",
                                         fp16=False)  # 使用 FP32 避免精度问题
                    
                    # 获取预测文本 - 处理不同的返回类型
                    if hasattr(result, 'text'):
                        # 如果是 DecodingResult 对象
                        hypothesis = result.text.strip()
                    elif isinstance(result, list):
                        # 如果是列表，取第一个元素
                        if result and hasattr(result[0], 'text'):
                            hypothesis = result[0].text.strip()
                        else:
                            hypothesis = ""
                    elif isinstance(result, dict) and "text" in result:
                        # 如果是字典，获取 text 字段
                        hypothesis = result["text"].strip()
                    else:
                        # 其他情况，转换为字符串
                        hypothesis = str(result).strip()
                    
                    # 移除可能的特殊标记
                    import re
                    hypothesis = re.sub(r'<\|.*?\|>', '', hypothesis).strip()
                    
                    # 将繁体中文转换为简体中文
                    hypothesis = traditional_to_simplified(hypothesis)
                    
                except Exception as e:
                    print(f"解码失败：{e}")
                    hypothesis = ""
                
                all_references.append(reference)
                all_hypotheses.append(hypothesis)
                
                # 调试输出：显示生成的文本
                print(f"\n参考文本：{reference}")
                print(f"预测文本：{hypothesis}")
                print(f"WER: {calculate_wer(reference, hypothesis):.2%}")
                print(f"CER: {calculate_cer(reference, hypothesis):.2%}\n")
            
            # 释放内存
            del mel, tokens, encoder_output, decoder_input, outputs, target, loss
            if device == "cuda":
                torch.cuda.empty_cache()
    
    # 计算平均损失
    if len(val_loader) > 0:
        avg_loss = total_loss / len(val_loader)
    else:
        avg_loss = 0.0
    
    # 计算 WER 和 CER
    if len(all_references) > 0:
        val_wer = sum(calculate_wer(r, h) for r, h in zip(all_references, all_hypotheses)) / len(all_references)
        val_cer = sum(calculate_cer(r, h) for r, h in zip(all_references, all_hypotheses)) / len(all_references)
    else:
        val_wer = 0.0
        val_cer = 0.0
    
    return avg_loss, val_wer, val_cer

def evaluate_final(model, val_loader, device):
    """
    最终评估
    
    Args:
        model: 模型
        val_loader: 验证数据加载器
        device: 设备
    
    Returns:
        final_wer: 最终 WER
        final_cer: 最终 CER
    """
    model.eval()
    all_references = []
    all_hypotheses = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="最终评估"):
            # 移动数据到设备
            mel = batch['mel'].to(device, non_blocking=True)
            texts = batch['text']
            
            # 生成预测（使用模型的解码方法）
            for i in range(mel.size(0)):
                # 提取单个样本
                mel_single = mel[i:i+1]
                
                # 获取参考文本
                reference = texts[i]
                
                # 使用模型的生成方法
                try:
                    result = model.decode(mel_single, 
                                       language="zh",
                                       task="transcribe",
                                       fp16=False)
                    
                    # 检查 result 的类型
                    if hasattr(result, 'text'):
                        # 如果是 DecodingResult 对象
                        hypothesis = result.text.strip()
                        
                        # 移除特殊标记
                        import re
                        hypothesis = re.sub(r'<\|.*?\|>', '', hypothesis).strip()
                    elif isinstance(result, list):
                        # 如果是列表，取第一个元素
                        if result:
                            hypothesis = str(result[0]).strip()
                        else:
                            hypothesis = ""
                    elif isinstance(result, dict) and "text" in result:
                        # 如果是字典，获取 text 字段
                        hypothesis = result["text"].strip()
                    else:
                        # 其他情况，转换为字符串
                        hypothesis = str(result).strip()
                except Exception as e:
                    print(f"解码失败: {e}")
                    hypothesis = ""
                
                all_references.append(reference)
                all_hypotheses.append(hypothesis)
            
            # 释放内存
            del mel
            if device == "cuda":
                torch.cuda.empty_cache()
    
    # 计算 WER 和 CER
    final_wer = sum(calculate_wer(r, h) for r, h in zip(all_references, all_hypotheses)) / len(all_references)
    final_cer = sum(calculate_cer(r, h) for r, h in zip(all_references, all_hypotheses)) / len(all_references)
    
    print(f"\n最终评估结果:")
    print(f"WER: {final_wer:.2%}")
    print(f"CER: {final_cer:.2%}")
    
    return final_wer, final_cer

def create_sample_metadata():
    """
    创建示例元数据
    """
    # 示例音频文件
    audio_files = [
        "21.wav",
        "22.wav",
        "(1).m4a",
        "(2).m4a",
        "23.wav",
        "31.wav"
    ]
    
    # 示例文本
    texts = [
        "院子门口不远处就是一个地铁站,这是一个美丽而神奇的景象。树上长满了又大又甜的桃子,海豚和鲸鱼的表演是很好看的节目,邮局门前的人行道上有一个蓝色的邮箱。",
        "他用画笔为自己画了一幅美丽的人生囊图，一只白鹿站在河畔的浅水里。宿舍楼旁边的十字路口有一个公共汽车站，夏日的夕阳很美丽，尤其是夏日大平原的夕阳。那个年代已经一去不复返了。",
        "风轻轻吹过窗边，阳光温柔地落在书页上，时光慢慢走，日子平淡又安稳。不必追赶什么，好好感受当下的每一刻，就已经很美好了。",
        "风轻轻吹过窗边，阳光透过树叶洒下斑驳的光影，日子慢下来的时候，才发现平凡的时光里，藏着最温柔的美好。不必追赶远方，珍惜当下，用心感受每一刻的温暖与安宁，就已经很幸福了。",
        "夏天有很多小朋友在沙滩上玩耍,校园依山恍惚风景如画,冬天北风呼啸雪花飞舞,忽然一道闪电把天空和大地都照亮了。古城门附近是一个休闲散心的好去处。",
        "天文望远镜可以用来观察天空，它到过很多地方观光旅游。山间的小道蜿蜒曲折，春天来了，山上开满了樱花，下雪以后田野里白皑皑的一片。",
    ]
    
    # 创建训练元数据 (4 个样本)
    train_metadata = []
    for i, (audio_file, text) in enumerate(zip(audio_files[:4], texts[:4])):
        if os.path.exists(audio_file):
            train_metadata.append({
                'audio': os.path.abspath(audio_file),
                'text': text
            })
        else:
            print(f"警告: 训练音频文件不存在: {audio_file}")
    
    # 创建验证元数据 (2 个样本)
    val_metadata = []
    for i, (audio_file, text) in enumerate(zip(audio_files[4:6], texts[4:6])):
        if os.path.exists(audio_file):
            val_metadata.append({
                'audio': os.path.abspath(audio_file),
                'text': text
            })
        else:
            print(f"警告: 验证音频文件不存在: {audio_file}")
    
    # 保存元数据到当前目录
    os.makedirs('data', exist_ok=True)
    with open('data/train_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(train_metadata, f, ensure_ascii=False, indent=2)
    
    with open('data/val_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(val_metadata, f, ensure_ascii=False, indent=2)
    
    print("示例元数据创建完成")
    print(f"训练样本: {len(train_metadata)}")
    print(f"验证样本: {len(val_metadata)}")

def main():
    """
    主函数
    """
    print("="*60)
    print("Whisper Large v3 增量训练")
    print("="*60)
    
    # 检查设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # 创建示例元数据
    print("\n创建示例元数据...")
    create_sample_metadata()
    
    # 创建数据加载器 (使用 batch_size=1 减少显存使用)
    print("\n创建数据加载器...")
    batch_size = 1  # 定义 batch_size
    train_loader, val_loader = create_data_loaders(
        'data/train_metadata.json',
        'data/val_metadata.json',
        batch_size=batch_size
    )
    
    # 训练参数
    epochs = 15  # 小数据集 (<10 小时) 适合 10-15 epochs
    save_dir = 'checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载模型 (使用 Large v3 模型)
    print("\n加载模型...")
    
    # 优先加载之前保存的模型
    model_loaded = False
    
    # 1. 尝试加载最终训练好的模型（15 轮训练完成后的模型）
    final_model_path = 'whisper_large_v3_finetuned.pth'
    if os.path.exists(final_model_path):
        print(f"发现之前训练完成的最终模型：{final_model_path}")
        try:
            model = whisper.load_model("large-v3", device="cpu")
            model = model.to("cpu")  # 先移动到 CPU
            model.load_state_dict(torch.load(final_model_path, map_location="cpu", weights_only=False))
            print("[OK] 成功加载最终训练模型！")
            model_loaded = True
        except Exception as e:
            print(f"加载最终模型失败：{e}，将尝试其他模型")
    
    # 2. 如果没有最终模型，尝试加载最新的检查点
    if not model_loaded:
        # 查找最新的检查点
        if os.path.exists(save_dir):
            checkpoints = [f for f in os.listdir(save_dir) if f.startswith('model_epoch_') and f.endswith('.pth')]
            if checkpoints:
                checkpoints.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
                latest_checkpoint = os.path.join(save_dir, checkpoints[-1])
                print(f"发现之前的训练检查点：{latest_checkpoint}")
                try:
                    model = whisper.load_model("large-v3", device="cpu")
                    model = model.to("cpu")
                    model.load_state_dict(torch.load(latest_checkpoint, map_location="cpu", weights_only=False))
                    print("[OK] 成功加载检查点模型！")
                    model_loaded = True
                except Exception as e:
                    print(f"加载检查点失败：{e}，将使用原始模型")
    
    # 3. 如果没有保存的模型，使用原始模型
    if not model_loaded:
        print("未找到之前训练的模型，加载原始 Whisper Large v3 模型...")
        try:
            model = whisper.load_model("large-v3", device="cpu")
            print("[OK] 模型加载成功！")
        except Exception as e:
            print(f"加载模型失败：{e}")
            # 尝试使用较小的模型
            print("尝试使用 medium 模型...")
            model = whisper.load_model("medium", device="cpu")
    
    # 移动到设备
    model = model.to(device)
    print(f"模型参数量：{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # 冻结层
    print("\n设置模型冻结层...")
    freeze_high_level(model)
    
    # 优化器和调度器
    print("\n设置优化器和调度器...")
    
    # 使用与原始模型相同的 weight_decay 参数
    weight_decay = 0.01  # Whisper 模型的默认 weight_decay
    
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-5,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=15,  # 与 epochs 对应
        eta_min=1e-6
    )
    
    # 多分类损失函数选择
    # CrossEntropyLoss 是最适合多分类任务的损失函数
    # ignore_index=-100 表示忽略标签为 -100 的样本
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
    
    # 开始训练
    print("\n" + "="*70)
    print("开始训练 Whisper Large v3")
    print("="*70)
    print(f"训练集大小：{len(train_loader.dataset)} 样本")
    print(f"验证集大小：{len(val_loader.dataset)} 样本")
    print(f"总 Epoch 数：{epochs}")
    print(f"批次大小：{batch_size}")
    print("="*70 + "\n")
    
    # 训练进度记录
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_wer': [],
        'val_cer': []
    }
    
    for epoch in range(epochs):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1:3d}/{epochs}")
        print(f"{'='*70}")
        
        # 训练
        print("训练阶段...", end="\r")
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # 验证
        print("验证阶段...", end="\r")
        val_loss, val_wer, val_cer = evaluate_epoch(model, val_loader, criterion, device)
        
        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_wer'].append(val_wer)
        history['val_cer'].append(val_cer)
        
        # 打印结果 - 使用表格格式
        print(f"\n{'训练结果:':<15}")
        print(f"  {'训练损失:':<12} {train_loss:>8.4f}")
        print(f"  {'验证损失:':<12} {val_loss:>8.4f}")
        print(f"  {'验证 WER:':<12} {val_wer:>8.2%}")
        print(f"  {'验证 CER:':<12} {val_cer:>8.2%}")
        print(f"  {'学习率:':<12} {current_lr:>8.2e}")
        
        # 保存检查点
        if (epoch + 1) % 5 == 0:
            save_path = os.path.join(save_dir, f'model_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), save_path)
            print(f"\n[OK] 检查点已保存：{save_path}")
    
    # 最终评估
    print(f"\n{'='*70}")
    print("最终评估")
    print(f"{'='*70}")
    evaluate_final(model, val_loader, device)
    
    # 保存最终模型
    final_save_path = 'whisper_large_v3_finetuned.pth'
    torch.save(model.state_dict(), final_save_path)
    print(f"\n[OK] 最终模型已保存：{final_save_path}")
    
    # 训练完成信息
    print(f"\n{'='*70}")
    print("训练完成！")
    print(f"{'='*70}")
    print(f"\n训练汇总:")
    print(f"  初始训练损失：{history['train_loss'][0]:.4f} -> 最终：{history['train_loss'][-1]:.4f}")
    print(f"  初始验证损失：{history['val_loss'][0]:.4f} -> 最终：{history['val_loss'][-1]:.4f}")
    print(f"  初始验证 WER:  {history['val_wer'][0]:.2%} -> 最终：{history['val_wer'][-1]:.2%}")
    print(f"  初始验证 CER:  {history['val_cer'][0]:.2%} -> 最终：{history['val_cer'][-1]:.2%}")
    
    # 找出最佳模型
    best_epoch = history['val_loss'].index(min(history['val_loss']))
    print(f"\n  最佳验证损失：{min(history['val_loss']):.4f} (Epoch {best_epoch+1})")
    print(f"  最佳验证 WER:  {min(history['val_wer']):.2%} (Epoch {history['val_wer'].index(min(history['val_wer']))+1})")
    print(f"  最佳验证 CER:  {min(history['val_cer']):.2%} (Epoch {history['val_cer'].index(min(history['val_cer']))+1})")
    
    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    main()
