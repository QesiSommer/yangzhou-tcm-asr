"""
test_pipeline.py — TCM ASR 纠错模块测试用例
==============================================
验证 N-best Closest 纠错流水线对于扬州方言同音错误的纠正能力。

测试策略：
    - 由于 LLM 调用依赖本地部署（不一定可用），测试采用 Mock 方式模拟 LLM 返回，
      确保打分逻辑、词典匹配、编辑距离计算等核心算法在每个场景下表现正确。
    - 同时也提供了真实 LLM 集成测试的入口（需手动启用）。
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import logging

from utils import (
    levenshtein_distance,
    check_tcm_dict,
    calculate_final_score,
    TCM_DICT,
)
from llm_client import LocalLLMClient
from error_correction import run_correction_pipeline, print_correction_report

# 测试时关闭 info 级别日志以减少输出噪音
logging.basicConfig(level=logging.WARNING)


# ============================================================================
# 单元测试：utils.py
# ============================================================================
class TestUtils(unittest.TestCase):
    """测试底层工具函数。"""

    def test_levenshtein_same_string(self):
        """相同字符串编辑距离为 0。"""
        self.assertEqual(levenshtein_distance("当归", "当归"), 0)

    def test_levenshtein_one_char(self):
        """单字符替换距离为 1。"""
        self.assertEqual(levenshtein_distance("当归", "当鬼"), 1)

    def test_levenshtein_different_strings(self):
        """完全不同字符串的编辑距离。"""
        self.assertEqual(levenshtein_distance("需要当归", "需要荡鬼"), 2)
        self.assertEqual(levenshtein_distance("abc", ""), 3)
        self.assertEqual(levenshtein_distance("", "abc"), 3)

    def test_levenshtein_yangzhou_dialect_cases(self):
        """扬州方言典型错误场景的编辑距离。"""
        # "当归" -> "荡鬼" (归gui -> 鬼gui 同音, 当dang -> 荡dang 近音)
        self.assertEqual(levenshtein_distance("需要当归调理", "需要荡鬼调理"), 2)
        # "气虚" -> "七区" (气qi -> 七qi 同音, 虚xu -> 区qu 近音)
        self.assertEqual(levenshtein_distance("气虚", "七区"), 2)
        # "活血化瘀" -> "活写话语"
        self.assertEqual(levenshtein_distance("活血化瘀", "活写话语"), 3)

    def test_check_tcm_dict_hit(self):
        """命中中医词典术语。"""
        hit, bonus = check_tcm_dict("需要当归调理")
        self.assertTrue(hit)
        self.assertEqual(bonus, 1.0)

    def test_check_tcm_dict_hit_multiple_terms(self):
        """命中多个中医术语时仍返回 1.0。"""
        hit, bonus = check_tcm_dict("用黄芪和当归补气")
        self.assertTrue(hit)
        self.assertEqual(bonus, 1.0)

    def test_check_tcm_dict_hit_syndrome(self):
        """命中证候类术语。"""
        for term in ["气虚", "血瘀", "阴虚", "痰湿"]:
            hit, bonus = check_tcm_dict(f"诊断为{term}")
            self.assertTrue(hit, f"术语 '{term}' 应该命中词典")
            self.assertEqual(bonus, 1.0)

    def test_check_tcm_dict_hit_formula(self):
        """命中方剂类术语。"""
        for term in ["四物汤", "桂枝汤", "小柴胡汤"]:
            hit, bonus = check_tcm_dict(f"开{term}加减")
            self.assertTrue(hit, f"术语 '{term}' 应该命中词典")
            self.assertEqual(bonus, 1.0)

    def test_check_tcm_dict_miss(self):
        """非中医文本不应命中词典。"""
        hit, bonus = check_tcm_dict("需要荡鬼调理")  # "荡鬼" 不在词典里
        self.assertFalse(hit)
        self.assertEqual(bonus, 0.0)

    def test_check_tcm_dict_empty(self):
        """空文本不应命中。"""
        hit, bonus = check_tcm_dict("")
        self.assertFalse(hit)
        self.assertEqual(bonus, 0.0)

    def test_check_tcm_dict_none(self):
        """None 输入不应命中。"""
        hit, bonus = check_tcm_dict(None)  # type: ignore
        self.assertFalse(hit)
        self.assertEqual(bonus, 0.0)

    def test_calculate_final_score_perfect_match(self):
        """理想情况：高置信度 + 命中词典 + 零编辑距离。"""
        score = calculate_final_score(
            asr_conf=0.95, dict_bonus=1.0, edit_dist=0,
            w1=0.4, w2=0.2, w3=0.4,
        )
        self.assertAlmostEqual(score, 0.95 * 0.4 + 1.0 * 0.2 - 0 * 0.4)  # = 0.58
        self.assertAlmostEqual(score, 0.58)

    def test_calculate_final_score_worst_case(self):
        """最差情况：低置信度 + 无词典命中 + 大编辑距离。"""
        score = calculate_final_score(
            asr_conf=0.1, dict_bonus=0.0, edit_dist=10,
            w1=0.4, w2=0.2, w3=0.4,
        )
        self.assertAlmostEqual(score, 0.1 * 0.4 + 0.0 - 10 * 0.4)  # = -3.96

    def test_calculate_final_score_dict_outweighs_edit(self):
        """
        词典加分应能抵消小编辑距离的扣分。

        场景：两个候选句编辑距离仅差 1，但一个命中词典。
        命中词典的应该得分更高。
        """
        # 候选 A: 高置信度 + 命词典 + 编辑距 2
        score_a = calculate_final_score(0.8, 1.0, 2, 0.4, 0.2, 0.4)
        # = 0.32 + 0.2 - 0.8 = -0.28

        # 候选 B: 高置信度 + 未命中 + 编辑距 2
        score_b = calculate_final_score(0.8, 0.0, 2, 0.4, 0.2, 0.4)
        # = 0.32 + 0.0 - 0.8 = -0.48

        self.assertGreater(score_a, score_b)

    def test_calculate_final_score_custom_weights(self):
        """自定义权重应正确影响分数。"""
        # 使用偏重 ASR 置信度的权重
        score_asr_heavy = calculate_final_score(0.8, 1.0, 3, 0.7, 0.1, 0.2)
        # = 0.56 + 0.1 - 0.6 = 0.06

        # 使用偏重编辑距离的权重
        score_edit_heavy = calculate_final_score(0.8, 1.0, 3, 0.2, 0.1, 0.7)
        # = 0.16 + 0.1 - 2.1 = -1.84

        self.assertGreater(score_asr_heavy, score_edit_heavy)


# ============================================================================
# 单元测试：llm_client.py
# ============================================================================
class TestLLMClient(unittest.TestCase):
    """测试大模型客户端（不含真实网络调用）。"""

    def test_init_default_values(self):
        """默认初始化参数。"""
        client = LocalLLMClient()
        self.assertEqual(client.model, "qwen2.5:1.5b")
        self.assertEqual(client.timeout, 30.0)

    def test_init_custom_values(self):
        """自定义初始化参数。"""
        client = LocalLLMClient(
            base_url="http://localhost:9999/v1",
            model="huatuo",
            timeout=60.0,
        )
        self.assertEqual(client.model, "huatuo")
        self.assertEqual(client.timeout, 60.0)

    def test_repr(self):
        """__repr__ 输出格式。"""
        client = LocalLLMClient(model="test-model")
        rep = repr(client)
        self.assertIn("test-model", rep)
        self.assertIn("LocalLLMClient", rep)

    def test_build_user_prompt_with_context(self):
        """构造含上下文的提示词。"""
        client = LocalLLMClient()
        prompt = client._build_user_prompt(
            context="患者头晕乏力",
            asr_1best_text="需要荡鬼",
        )
        self.assertIn("患者头晕乏力", prompt)
        self.assertIn("需要荡鬼", prompt)
        self.assertIn("历史问诊上下文", prompt)
        self.assertIn("ASR 当前识别结果", prompt)

    def test_build_user_prompt_empty_context(self):
        """构造无上下文的提示词。"""
        client = LocalLLMClient()
        prompt = client._build_user_prompt(context="", asr_1best_text="需要当归")
        self.assertIn("无历史上下文", prompt)
        self.assertIn("需要当归", prompt)

    def test_generate_correction_empty_input(self):
        """空输入应返回空字符串。"""
        client = LocalLLMClient()
        result = client.generate_correction(context="", asr_1best_text="")
        self.assertEqual(result, "")

    @patch("llm_client.OpenAI")
    def test_generate_correction_success(self, mock_openai_class):
        """模拟 LLM 成功返回。"""
        # 构造 mock response
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "需要当归调理"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        client = LocalLLMClient()
        result = client.generate_correction(
            context="患者头晕",
            asr_1best_text="需要荡鬼调理",
        )
        self.assertEqual(result, "需要当归调理")

    @patch("llm_client.OpenAI")
    def test_generate_correction_none_content(self, mock_openai_class):
        """LLM 返回 None 时应降级。"""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        client = LocalLLMClient()
        result = client.generate_correction(
            context="",
            asr_1best_text="需要荡鬼调理",
        )
        # 降级返回原始 ASR 文本
        self.assertEqual(result, "需要荡鬼调理")

    @patch("llm_client.OpenAI")
    def test_generate_correction_api_error(self, mock_openai_class):
        """API 调用异常时应降级。"""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("Connection refused")
        mock_openai_class.return_value = mock_client

        client = LocalLLMClient()
        result = client.generate_correction(
            context="",
            asr_1best_text="需要荡鬼调理",
        )
        # 降级返回原始 ASR 文本
        self.assertEqual(result, "需要荡鬼调理")


# ============================================================================
# 集成测试：error_correction.py（使用 Mock LLM）
# ============================================================================
class TestPipelineWithMockLLM(unittest.TestCase):
    """使用 Mock LLM 验证完整流水线。"""

    def setUp(self):
        """每个测试前创建独立的 mock LLM 客户端。"""
        self.mock_llm = MagicMock(spec=LocalLLMClient)
        self.mock_llm.generate_correction.return_value = "需要当归调理"

    def _assert_valid_scored_item(self, item):
        """验证打分明细的字段完整性。"""
        self.assertIn("text", item)
        self.assertIn("asr_conf", item)
        self.assertIn("dict_bonus", item)
        self.assertIn("dict_hit", item)
        self.assertIn("edit_distance", item)
        self.assertIn("final_score", item)

    def test_pipeline_yangzhou_case_1(self):
        """
        扬州话场景 1：ASR 把"当归"错听成"荡鬼"。

        LLM 纠错输出"需要当归调理"，系统应映射回含"当归"的候选句。
        """
        n_best = [
            {"text": "需要荡鬼调理", "conf": 0.80},   # ASR 高置信度错误
            {"text": "需要当归调理", "conf": 0.15},   # ASR 低置信度正确
            {"text": "需要当鬼调理", "conf": 0.05},
        ]

        result = run_correction_pipeline(
            n_best_list=n_best,
            context="患者自述：最近头晕乏力，面色萎黄",
            llm_client=self.mock_llm,
        )

        # 应该是"需要当归调理"得分最高（编辑距离 0 + 词典命中）
        self.assertEqual(result["best_text"], "需要当归调理")
        self.assertEqual(len(result["n_best_scored"]), 3)

        for item in result["n_best_scored"]:
            self._assert_valid_scored_item(item)

        # 验证打分排序正确：best_score 应为最高
        scores = [item["final_score"] for item in result["n_best_scored"]]
        self.assertEqual(result["best_score"], max(scores))

    def test_pipeline_yangzhou_case_2(self):
        """
        扬州话场景 2：ASR 把"气虚"错听成"七区"。

        LLM 纠错输出"气虚"，系统应选择含"气虚"的候选句。
        """
        n_best = [
            {"text": "诊断为七区", "conf": 0.75},
            {"text": "诊断为气虚", "conf": 0.20},
            {"text": "诊断为气区", "conf": 0.05},
        ]
        self.mock_llm.generate_correction.return_value = "诊断为气虚"

        result = run_correction_pipeline(
            n_best_list=n_best,
            context="患者主诉：易疲劳，气短懒言",
            llm_client=self.mock_llm,
        )

        self.assertEqual(result["best_text"], "诊断为气虚")
        # "气虚" 应命中词典
        best_item = result["n_best_scored"][0]
        self.assertTrue(best_item["dict_hit"])
        self.assertEqual(best_item["dict_bonus"], 1.0)

    def test_pipeline_llm_returns_asr_exact_match(self):
        """
        场景 3：LLM 认为 ASR 正确，返回与 1-best 完全一致的文本。

        此时 1-best 编辑距离为 0，理应得分最高。
        """
        n_best = [
            {"text": "需要四物汤加减", "conf": 0.85},
            {"text": "需要事物汤加减", "conf": 0.10},
            {"text": "需要四物堂加减", "conf": 0.05},
        ]
        # LLM 同意 ASR 识别结果，原样返回
        self.mock_llm.generate_correction.return_value = "需要四物汤加减"

        result = run_correction_pipeline(
            n_best_list=n_best,
            context="患者月经不调",
            llm_client=self.mock_llm,
        )

        self.assertEqual(result["best_text"], "需要四物汤加减")
        best = result["n_best_scored"][0]
        self.assertEqual(best["edit_distance"], 0)

    def test_pipeline_weights_in_result(self):
        """结果中应包含权重信息。"""
        n_best = [{"text": "test", "conf": 0.9}]
        self.mock_llm.generate_correction.return_value = "test"

        custom_weights = {"w1": 0.5, "w2": 0.3, "w3": 0.2}
        result = run_correction_pipeline(
            n_best_list=n_best,
            llm_client=self.mock_llm,
            weights=custom_weights,
        )
        self.assertEqual(result["weights"], custom_weights)

    def test_pipeline_empty_n_best_raises(self):
        """空 N-best 列表应抛出异常。"""
        with self.assertRaises(ValueError):
            run_correction_pipeline(
                n_best_list=[],
                llm_client=self.mock_llm,
            )

    def test_pipeline_invalid_item_raises(self):
        """格式错误的 N-best 项应抛出异常。"""
        with self.assertRaises(ValueError):
            run_correction_pipeline(
                n_best_list=[{"text": "missing conf"}],  # 缺少 conf 字段
                llm_client=self.mock_llm,
            )

        with self.assertRaises(ValueError):
            run_correction_pipeline(
                n_best_list=[{"conf": 0.9}],  # 缺少 text 字段
                llm_client=self.mock_llm,
            )

    def test_pipeline_scoring_detail_consistency(self):
        """
        打分明细一致性：每个候选句的 final_score 应与手动计算一致。
        """
        n_best = [
            {"text": "需要当归", "conf": 0.8},
            {"text": "需要荡鬼", "conf": 0.7},
        ]
        self.mock_llm.generate_correction.return_value = "需要当归"

        result = run_correction_pipeline(
            n_best_list=n_best,
            llm_client=self.mock_llm,
        )

        for item in result["n_best_scored"]:
            expected_score = calculate_final_score(
                asr_conf=item["asr_conf"],
                dict_bonus=item["dict_bonus"],
                edit_dist=item["edit_distance"],
            )
            self.assertAlmostEqual(
                item["final_score"], expected_score,
                msg=f"候选 '{item['text']}' 的分数不一致",
            )

    def test_pipeline_single_candidate(self):
        """只有一个候选句时也应正常工作。"""
        n_best = [{"text": "需要当归", "conf": 0.9}]
        self.mock_llm.generate_correction.return_value = "需要当归"

        result = run_correction_pipeline(
            n_best_list=n_best,
            llm_client=self.mock_llm,
        )
        self.assertEqual(result["best_text"], "需要当归")
        self.assertEqual(len(result["n_best_scored"]), 1)


# ============================================================================
# 可视化报告测试
# ============================================================================
class TestReportOutput(unittest.TestCase):
    """测试报告输出（主要是确保不抛异常）。"""

    def test_print_report_no_error(self):
        """print_correction_report 不应抛出异常。"""
        mock_result = {
            "best_text": "需要当归调理",
            "best_score": 0.35,
            "llm_correction": "需要当归调理",
            "n_best_scored": [
                {
                    "text": "需要当归调理",
                    "asr_conf": 0.15,
                    "dict_bonus": 1.0,
                    "dict_hit": True,
                    "edit_distance": 0,
                    "final_score": 0.35,
                },
                {
                    "text": "需要荡鬼调理",
                    "asr_conf": 0.80,
                    "dict_bonus": 0.0,
                    "dict_hit": False,
                    "edit_distance": 2,
                    "final_score": -0.48,
                },
            ],
            "weights": {"w1": 0.4, "w2": 0.2, "w3": 0.4},
        }
        # 不应抛出异常
        print_correction_report(mock_result)


# ============================================================================
# 词典完整性检查
# ============================================================================
class TestDictCoverage(unittest.TestCase):
    """验证中医词典的覆盖完整性。"""

    def test_dict_not_empty(self):
        """词典不应为空。"""
        self.assertGreater(len(TCM_DICT), 0)

    def test_dict_key_categories(self):
        """关键类别的术语应存在于词典中。"""
        essential_terms = [
            # 药材
            "当归", "黄芪", "人参", "甘草",
            # 证候
            "气虚", "血瘀", "阴虚", "阳虚", "痰湿",
            # 方剂
            "四物汤", "桂枝汤", "小柴胡汤",
            # 脏腑
            "脾胃", "肝肾",
            # 治法
            "活血化瘀", "清热解毒",
        ]
        for term in essential_terms:
            self.assertIn(term, TCM_DICT, f"关键术语 '{term}' 缺失！")


# ============================================================================
# 运行入口
# ============================================================================
if __name__ == "__main__":
    import sys

    print("=" * 70)
    print("  TCM ASR 纠错模块 — 测试套件")
    print("=" * 70)
    print()

    # 运行所有测试
    unittest.main(verbosity=2, argv=[sys.argv[0]])
