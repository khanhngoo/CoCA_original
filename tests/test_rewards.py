import unittest

from coca.rewards import compute_group_rewards, extract_final_answer, normalize_advantages


class RewardTest(unittest.TestCase):
    def test_extract_boxed_answer(self):
        self.assertEqual(extract_final_answer("So the result is \\boxed{42}."), "42")

    def test_group_rewards_use_gesr(self):
        rewards = compute_group_rewards([0.0, 0.5, 1.0, None], [False, True, True, False])
        self.assertEqual(rewards.gesr, 0.5)
        self.assertEqual(rewards.answer_rewards, [0.0, 1.0, 1.0, 0.0])
        self.assertAlmostEqual(rewards.confidence_rewards[1], -0.0)
        self.assertEqual(rewards.confidence_rewards[3], -1.0)

    def test_advantages_zero_when_constant(self):
        self.assertEqual(normalize_advantages([1.0, 1.0, 1.0]), [0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()

