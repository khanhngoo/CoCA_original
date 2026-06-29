import unittest

from coca.data import (
    EVAL_BENCHMARKS,
    extract_gold_answer,
    normalize_eval_row,
    normalize_example,
)


class DataTest(unittest.TestCase):
    def test_normalize_big_math_schema(self):
        example = normalize_example(
            {
                "problem": "What is 2+2?",
                "answer": "4",
                "source": "unit",
                "domain": "arithmetic",
                "llama8b_solve_rate": 1.0,
            }
        )
        self.assertEqual(example.problem, "What is 2+2?")
        self.assertEqual(example.answer, "4")
        self.assertEqual(example.source, "unit")
        self.assertEqual(example.metadata["llama8b_solve_rate"], 1.0)


class EvalDatasetNormalizationTest(unittest.TestCase):
    def test_gsm8k_extracts_answer_after_marker(self):
        self.assertEqual(extract_gold_answer("steps...\n#### 1,024", "gsm8k"), "1024")

    def test_plain_answer_passthrough(self):
        self.assertEqual(extract_gold_answer("  42 ", "plain"), "42")

    def test_normalize_gsm8k_row(self):
        example = normalize_eval_row(
            {"question": "Q?", "answer": "work\n#### 7"},
            EVAL_BENCHMARKS["gsm8k"],
        )
        self.assertEqual(example.problem, "Q?")
        self.assertEqual(example.answer, "7")
        self.assertEqual(example.source, "openai/gsm8k")

    def test_normalize_math500_row(self):
        example = normalize_eval_row(
            {"problem": "Solve x", "answer": "\\frac{1}{2}"},
            EVAL_BENCHMARKS["math500"],
        )
        self.assertEqual(example.problem, "Solve x")
        self.assertEqual(example.answer, "\\frac{1}{2}")

    def test_missing_key_raises(self):
        with self.assertRaises(KeyError):
            normalize_eval_row({"answer": "7"}, EVAL_BENCHMARKS["gsm8k"])


if __name__ == "__main__":
    unittest.main()
