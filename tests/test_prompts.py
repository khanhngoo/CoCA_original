import unittest

from coca.prompts import (
    DEFAULT_SYSTEM_PROMPT,
    USER_PROMPT_SUFFIX,
    build_messages,
    parse_confidence,
    segment_completion_token_masks,
    split_confidence_answer,
    tokens_to_confidence_close,
)


class CharTokenizer:
    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(item) for item in ids)

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        payload = {"input_ids": [ord(char) for char in text]}
        if return_offsets_mapping:
            payload["offset_mapping"] = [(idx, idx + 1) for idx in range(len(text))]
        return payload


class PromptParsingTest(unittest.TestCase):
    def test_parse_decimal_confidence(self):
        self.assertEqual(parse_confidence("<confidence>0.75</confidence> answer"), 0.75)

    def test_parse_percent_confidence(self):
        self.assertEqual(parse_confidence("<confidence>75%</confidence> answer"), 0.75)

    def test_parse_integer_percent_like_confidence(self):
        self.assertEqual(parse_confidence("<confidence>75</confidence> answer"), 0.75)

    def test_reject_out_of_range_confidence(self):
        self.assertIsNone(parse_confidence("<confidence>125%</confidence> answer"))

    def test_split_confidence_answer(self):
        raw, answer = split_confidence_answer("<confidence>0.1</confidence>\n42")
        self.assertEqual(raw, "0.1")
        self.assertEqual(answer, "42")

    def test_segment_completion_token_masks(self):
        text = "<confidence>0.5</confidence> answer"
        masks = segment_completion_token_masks(CharTokenizer(), [ord(char) for char in text])
        self.assertTrue(any(masks.confidence))
        self.assertTrue(any(masks.answer))
        close_index = text.index("</confidence>") + len("</confidence>")
        self.assertTrue(all(masks.confidence[:close_index]))
        self.assertTrue(all(masks.answer[close_index:]))


class PaperPromptFidelityTest(unittest.TestCase):
    def test_system_prompt_matches_paper_wording(self):
        self.assertIn("confidence level", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("<confidence> </confidence>", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("between 0 and 1", DEFAULT_SYSTEM_PROMPT)

    def test_user_message_has_boxed_suffix(self):
        messages = build_messages("What is 2+2?")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("\\boxed{}", messages[1]["content"])
        self.assertTrue(messages[1]["content"].startswith("What is 2+2?"))
        self.assertIn("\\boxed{}", USER_PROMPT_SUFFIX)

    def test_tokens_to_confidence_close_returns_int(self):
        text = "<confidence>0.5</confidence> rest of answer"
        ttc = tokens_to_confidence_close(text, CharTokenizer())
        self.assertIsInstance(ttc, int)
        self.assertEqual(ttc, text.index("</confidence>") + len("</confidence>"))

    def test_tokens_to_confidence_close_none_without_tag(self):
        self.assertIsNone(tokens_to_confidence_close("no tag here", CharTokenizer()))


if __name__ == "__main__":
    unittest.main()
