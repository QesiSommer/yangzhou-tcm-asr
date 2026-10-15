"""
llm_client.py — 本地大模型连接层
=================================
封装与本地部署的中医大模型（vLLM / OpenAI 兼容 API）的通信逻辑。
"""

import logging
from typing import Optional

from openai import OpenAI

logger = logging.getLogger(__name__)


class LocalLLMClient:
    """
    本地大模型客户端，用于调用兼容 OpenAI API 格式的 vLLM 接口。

    封装了与中医纠错大模型的通信，包括：
    - 连接管理
    - Prompt 构造
    - 降级处理（网络故障时返回原始文本）

    Usage:
        client = LocalLLMClient()
        corrected = client.generate_correction(
            context="患者自述头晕乏力",
            asr_1best_text="需要荡鬼调理"
        )
    """

    # 系统提示词：定义大模型角色与行为边界
    SYSTEM_PROMPT = (
        "你是一名资深的中医语音识别纠错专家。"
        "你的任务是根据患者的中医问诊上下文，将前端语音识别（ASR）可能存在的同音/近音错误进行纠正。"
        "\n\n"
        "【重要规则】\n"
        "1. 仅输出纠正后的一个完整句子，不要添加任何解释、评论或标注。\n"
        "2. 如果 ASR 识别的文本没有明显错误，请原样输出。\n"
        "3. 严禁扩写、编造或添加 ASR 文本中不存在的信息。"
        "在医疗场景中，任何未经证实的添加都可能造成严重后果。\n"
        "4. 特别关注扬州方言中常见的同音替换错误，"
        '如"当归"误为"荡鬼"、"气虚"误为"七区"等。\n'
        "5. 纠正时优先考虑中医专业术语的合理性。"
    )

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        model: str = "qwen2.5:1.5b",
        timeout: float = 30.0,
        api_key: str = "not-needed",
    ):
        """
        初始化本地 LLM 客户端。

        Args:
            base_url: 本地大模型的 API 地址（OpenAI 兼容格式）。
            model: 模型名称/标识。
            timeout: 请求超时时间（秒）。
            api_key: API 密钥（本地部署通常无需真实密钥，设占位值即可）。
        """
        self.model = model
        self.timeout = timeout

        self._client = OpenAI(
            base_url=base_url.rstrip("/"),
            api_key=api_key,
            timeout=timeout,
        )
        logger.info("LocalLLMClient 已初始化: base_url=%s, model=%s", base_url, model)

    def _build_user_prompt(self, context: str, asr_1best_text: str) -> str:
        """
        构造发送给大模型的用户消息。

        Args:
            context: 历史对话/问诊上下文。
            asr_1best_text: ASR 最置信候选句。

        Returns:
            格式化的用户提示词字符串。
        """
        context_block = context if context else "（无历史上下文）"
        return (
            f"【历史问诊上下文】\n{context_block}\n\n"
            f"【ASR 当前识别结果（可能含同音错误）】\n{asr_1best_text}\n\n"
            f"请输出纠正后的正确文本："
        )

    def generate_correction(
        self,
        context: str,
        asr_1best_text: str,
    ) -> str:
        """
        调用本地大模型对 ASR 文本进行语义纠错。

        将历史上下文和当前 ASR 最高置信度候选句发送给中医纠错大模型，
        获取其自由生成的纠错文本。

        Args:
            context: 历史对话上下文（可为空字符串）。
            asr_1best_text: 当前轮 ASR 的最高分候选句。

        Returns:
            大模型纠错后的文本。若请求失败或结果为空，则降级返回原始 asr_1best_text。
        """
        if not asr_1best_text:
            logger.warning("asr_1best_text 为空，返回空字符串")
            return ""

        user_prompt = self._build_user_prompt(context, asr_1best_text)

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,  # 低温度，确保输出稳定可预测
                max_tokens=256,   # 限制输出长度，防止扩写
                timeout=self.timeout,
            )

            corrected_text = response.choices[0].message.content
            if corrected_text is None:
                logger.warning("LLM 返回了空内容 (None)，降级使用原始 ASR 文本")
                return asr_1best_text

            corrected_text = corrected_text.strip()
            if not corrected_text:
                logger.warning("LLM 返回了空白内容，降级使用原始 ASR 文本")
                return asr_1best_text

            logger.info(
                "LLM 纠错完成: '%s...' -> '%s...'",
                asr_1best_text[:30], corrected_text[:30],
            )
            return corrected_text

        except Exception as e:
            logger.warning(
                "LLM 请求失败 (%s: %s)，降级使用原始 ASR 文本",
                type(e).__name__, e,
            )
            return asr_1best_text

    def __repr__(self) -> str:
        return f"LocalLLMClient(model={self.model!r}, timeout={self.timeout})"
