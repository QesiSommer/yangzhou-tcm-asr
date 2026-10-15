"""
error_correction.py — TCM ASR 纠错核心流水线
==============================================
整合 utils.py（词典/打分）和 llm_client.py（大模型连接），
实现完整的 N-best Closest 约束纠错流程。

整体流程：
    1. 接收 ASR N-best 候选列表
    2. 调用 LLM 获取自由纠错文本
    3. 对每个候选句：计算词典加分 + 编辑距离 + 综合得分
    4. 返回得分最高的候选句（保证输出是 ASR 物理路径内的文本）
"""

import logging
from typing import Any

from utils import levenshtein_distance, check_tcm_dict, calculate_final_score
from llm_client import LocalLLMClient

logger = logging.getLogger(__name__)

# ============================================================================
# 默认权重配置（可调参）
# ============================================================================
DEFAULT_WEIGHTS = {
    "w1": 0.4,   # ASR 置信度权重
    "w2": 0.2,   # 词典加分权重
    "w3": 0.4,   # 编辑距离扣分权重
}


def run_correction_pipeline(
    n_best_list: list[dict[str, Any]],
    context: str = "",
    llm_client: LocalLLMClient | None = None,
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    执行 N-best Closest 纠错流水线，返回最佳候选句及详细打分明细。

    Pipeline:
        1. 取出 N-best 列表中 ASR 置信度最高的句子（1-best）
        2. 将其与上下文一同发送给本地大模型，获取 LLM 自由生成的纠错文本
        3. 对 N-best 中的每一项候选句：
            a. 计算词典加分（是否命中中医术语）
            b. 计算与 LLM 纠错文本的编辑距离
            c. 带入综合打分公式算出 Final_Score
        4. 按 Final_Score 降序排列，返回最高分的候选句

    Args:
        n_best_list: ASR N-best 候选列表，每项为 {"text": str, "conf": float}。
                     例如: [{"text": "需要荡鬼", "conf": 0.8}, {"text": "需要当归", "conf": 0.2}]
        context: 历史问诊上下文（可为空字符串）。
        llm_client: 本地大模型客户端实例。若为 None，则自动创建默认实例。
        weights: 权重字典 {"w1": ..., "w2": ..., "w3": ...}。若为 None，使用默认值。

    Returns:
        包含最终结果和打分明细的字典：
        {
            "best_text": str,           # 最终选中的最佳候选句
            "best_score": float,         # 最佳候选句的综合得分
            "llm_correction": str,       # LLM 自由生成的纠错文本
            "n_best_scored": list[dict], # 每个候选句的详细打分
            "weights": dict,             # 实际使用的权重
        }

    Raises:
        ValueError: 当 n_best_list 为空或格式不正确时。

    Example:
        >>> client = LocalLLMClient(model="shennong")
        >>> result = run_correction_pipeline(
        ...     n_best_list=[
        ...         {"text": "需要荡鬼", "conf": 0.8},
        ...         {"text": "需要当归", "conf": 0.2},
        ...     ],
        ...     context="患者自述头晕乏力",
        ...     llm_client=client,
        ... )
        >>> print(result["best_text"])
        '需要当归'
    """
    # ---- 输入校验 ----
    if not n_best_list:
        raise ValueError("n_best_list 不能为空")

    for i, item in enumerate(n_best_list):
        if "text" not in item or "conf" not in item:
            raise ValueError(
                f"n_best_list[{i}] 不合法，每项必须包含 'text' 和 'conf' 字段: {item}"
            )

    # ---- 权重配置 ----
    w = weights if weights is not None else DEFAULT_WEIGHTS
    w1, w2, w3 = w["w1"], w["w2"], w["w3"]

    # ---- 初始化 LLM 客户端 ----
    if llm_client is None:
        llm_client = LocalLLMClient()

    # ---- 取出 ASR 最置信候选句（1-best） ----
    # 按 conf 降序排列，取第一个
    sorted_by_conf = sorted(n_best_list, key=lambda x: x["conf"], reverse=True)
    asr_1best_text = sorted_by_conf[0]["text"]
    logger.info(f"ASR 1-best 候选句: '{asr_1best_text}' (conf={sorted_by_conf[0]['conf']})")

    # ---- 步骤 2: 调用大模型获取自由纠错文本 ----
    llm_correction = llm_client.generate_correction(
        context=context,
        asr_1best_text=asr_1best_text,
    )
    logger.info(f"LLM 纠错输出: '{llm_correction}'")

    # ---- 步骤 3: 对每个候选句计算综合得分 ----
    n_best_scored: list[dict[str, Any]] = []
    for item in n_best_list:
        candidate_text: str = item["text"]
        asr_conf: float = item["conf"]

        # 3a. 词典加分
        hit_flag, dict_bonus = check_tcm_dict(candidate_text)

        # 3b. 计算与 LLM 纠错文本的编辑距离
        edit_dist = levenshtein_distance(llm_correction, candidate_text)

        # 3c. 综合打分
        final_score = calculate_final_score(
            asr_conf=asr_conf,
            dict_bonus=dict_bonus,
            edit_dist=edit_dist,
            w1=w1,
            w2=w2,
            w3=w3,
        )

        scored_item = {
            "text": candidate_text,
            "asr_conf": asr_conf,
            "dict_bonus": dict_bonus,
            "dict_hit": hit_flag,
            "edit_distance": edit_dist,
            "final_score": final_score,
        }
        n_best_scored.append(scored_item)
        logger.debug(
            f"候选句打分: {candidate_text[:20]:<20s} "
            f"ASR={asr_conf:.3f} DictBonus={dict_bonus:.1f} "
            f"EditDist={edit_dist:2d} Final={final_score:.4f}"
        )

    # ---- 步骤 4: 按 Final_Score 降序排列 ----
    n_best_scored.sort(key=lambda x: x["final_score"], reverse=True)

    best = n_best_scored[0]
    logger.info(
        f"纠错完成: ASR 1-best='{asr_1best_text}' "
        f"-> LLM='{llm_correction}' "
        f"-> 最终输出='{best['text']}' (score={best['final_score']:.4f})"
    )

    return {
        "best_text": best["text"],
        "best_score": best["final_score"],
        "llm_correction": llm_correction,
        "n_best_scored": n_best_scored,
        "weights": dict(w),
    }


# ============================================================================
# 便捷函数：格式化输出纠错结果
# ============================================================================
def print_correction_report(result: dict[str, Any]) -> None:
    """
    以可读格式打印纠错结果报告。

    Args:
        result: run_correction_pipeline 的返回值。
    """
    print("\n" + "=" * 70)
    print("  TCM ASR 纠错报告")
    print("=" * 70)
    print(f"  LLM 自由纠错文本   : {result['llm_correction']}")
    print(f"  最终输出           : {result['best_text']}")
    print(f"  最终得分           : {result['best_score']:.4f}")
    print(f"  权重配置           : w1={result['weights']['w1']}, "
          f"w2={result['weights']['w2']}, w3={result['weights']['w3']}")
    print("-" * 70)
    print(f"  {'候选句':<20s} {'ASR':>6s} {'词典':>4s} {'编辑距':>6s} {'最终分':>8s}")
    print("-" * 70)
    for item in result["n_best_scored"]:
        hit_marker = "[Y]" if item["dict_hit"] else "[N]"
        print(
            f"  {item['text']:<20s} "
            f"{item['asr_conf']:>6.3f} "
            f"{hit_marker:>4s} "
            f"{item['edit_distance']:>6d} "
            f"{item['final_score']:>8.4f}"
        )
    print("=" * 70 + "\n")


# ============================================================================
# CLI 入口
# ============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    # 示例：模拟一次纠错流程
    print("=" * 70)
    print("  TCM ASR 纠错模块 — 流水线演示")
    print("=" * 70)

    # Mock 数据
    mock_n_best = [
        {"text": "需要荡鬼调理", "conf": 0.80},
        {"text": "需要当归调理", "conf": 0.15},
        {"text": "需要当鬼调理", "conf": 0.05},
    ]
    mock_context = "患者自述：最近头晕乏力，面色萎黄，经量少色淡。"

    print(f"\n输入 N-best 候选:")
    for item in mock_n_best:
        print(f"  - '{item['text']}' (conf={item['conf']})")
    print(f"上下文: {mock_context}\n")

    try:
        result = run_correction_pipeline(
            n_best_list=mock_n_best,
            context=mock_context,
        )
        print_correction_report(result)
    except Exception as e:
        # LLM 不可用时也能演示打分配置（使用降级输出 = 原始 ASR 文本）
        logger.error(f"流水线执行异常: {e}")
        print(f"\n[注意] 本地大模型不可用时，LLM 降级返回原始 ASR 1-best 文本。")
        print(f"此时编辑距离计算仍可验证，但语义纠错效果将不明显。\n")
