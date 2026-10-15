"""
utils.py — TCM ASR 纠错模块底层工具库
==========================================
提供编辑距离计算、中医词典匹配、综合打分等核心工具函数。
"""

from typing import Tuple

try:
    from Levenshtein import distance as lev_distance
except ImportError:
    # 降级方案：如果 python-Levenshtein 未安装，使用纯 Python 实现
    def lev_distance(s1: str, s2: str) -> int:
        """纯 Python 实现的编辑距离（降级方案）。"""
        if len(s1) < len(s2):
            s1, s2 = s2, s1
        if len(s2) == 0:
            return len(s1)

        prev_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                # 插入、删除、替换三种操作的代价
                insertions = prev_row[j + 1] + 1
                deletions = curr_row[j] + 1
                substitutions = prev_row[j] + (0 if c1 == c2 else 1)
                curr_row.append(min(insertions, deletions, substitutions))
            prev_row = curr_row
        return prev_row[-1]


# ============================================================================
# 中医词典 (Mock 数据)
# ============================================================================
TCM_DICT: set[str] = {
    # --- 药材类 (Herbs) ---
    "当归", "人参", "黄芪", "甘草", "白术", "茯苓", "川芎", "白芍",
    "熟地黄", "枸杞", "陈皮", "半夏", "丹参", "红花", "桃仁", "大黄",
    "黄连", "黄芩", "柴胡", "桂枝", "附子", "干姜", "肉桂", "山药",
    "麦冬", "天麻", "杜仲", "牛膝", "桔梗", "杏仁", "桑白皮", "金银花",
    # --- 证候/病机类 (Syndromes & Pathomechanisms) ---
    "气虚", "血瘀", "气滞", "血虚", "阴虚", "阳虚", "痰湿", "湿热",
    "寒凝", "风邪", "气逆", "津亏", "营卫不和",
    # --- 脏腑/经络类 (Organs & Meridians) ---
    "经络", "针灸", "脾胃", "肝肾", "心肺", "督脉", "任脉", "腧穴",
    "足三里", "合谷", "内关",
    # --- 治法/方剂类 (Treatment Methods & Formulas) ---
    "补中益气", "活血化瘀", "清热解毒", "祛风散寒", "温经通络",
    "四物汤", "桂枝汤", "小柴胡汤", "归脾汤", "六味地黄丸",
}


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    计算两个字符串之间的编辑距离（Levenshtein Distance）。

    编辑距离定义为将 s1 转换为 s2 所需的最少单字符编辑操作次数
    （插入、删除、替换）。

    Args:
        s1: 第一个字符串（通常是 LLM 纠错文本）。
        s2: 第二个字符串（通常是 ASR 候选句）。

    Returns:
        两个字符串之间的编辑距离（int）。

    Example:
        >>> levenshtein_distance("需要当归", "需要荡鬼")
        2
    """
    return lev_distance(s1, s2)


def check_tcm_dict(text: str) -> Tuple[bool, float]:
    """
    检查文本中是否包含中医专业术语，并返回词典加分。

    扫描输入文本，判断是否命中预设的 TCM 词典中的任何词条。
    命中得 1.0 分（bonus），未命中得 0。

    Args:
        text: 待检查的文本字符串。

    Returns:
        (hit_flag, bonus_score) 元组：
        - hit_flag (bool): 是否命中词典。
        - bonus_score (float): 词典加分（1.0 或 0）。

    Example:
        >>> check_tcm_dict("需要当归")
        (True, 1.0)
        >>> check_tcm_dict("需要荡鬼")
        (False, 0.0)
    """
    if not text or not isinstance(text, str):
        return False, 0.0

    for term in TCM_DICT:
        if term in text:
            return True, 1.0

    return False, 0.0


def calculate_final_score(
    asr_conf: float,
    dict_bonus: float,
    edit_dist: int,
    w1: float = 0.4,
    w2: float = 0.2,
    w3: float = 0.4,
) -> float:
    """
    计算候选句的综合最终得分。

    打分公式：
        Final_Score = (ASR_Conf * w1) + (Dict_Bonus * w2) - (Levenshtein_Dist * w3)

    分数越高，表示该候选句越可能是正确的 ASR 识别结果。

    Args:
        asr_conf: ASR 引擎给出的原始置信度（0~1）。
        dict_bonus: 词典匹配加分（命中 = 1.0，未命中 = 0）。
        edit_dist: 候选句与 LLM 纠错文本之间的编辑距离。
        w1: ASR 置信度的权重因子（默认 0.4）。
        w2: 词典加分的权重因子（默认 0.2）。
        w3: 编辑距离扣分的权重因子（默认 0.4）。

    Returns:
        综合得分（float）。得分范围理论上为 (-∞, w1 + w2]，实际中非负居多。

    Example:
        >>> calculate_final_score(0.8, 1.0, 2, 0.4, 0.2, 0.4)
        0.32 + 0.2 - 0.8 = -0.28
        >>> calculate_final_score(0.8, 0.0, 2, 0.4, 0.2, 0.4)
        0.32 + 0.0 - 0.8 = -0.48
    """
    return (asr_conf * w1) + (dict_bonus * w2) - (edit_dist * w3)
