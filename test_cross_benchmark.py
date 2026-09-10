import copy
import unittest

from cross_benchmark import trial_messages


class TrialMessagesTest(unittest.TestCase):
    def test_existing_system_is_preserved_without_duplicate_role(self):
        original = [{"role": "system", "content": "Write runnable code."},
                    {"role": "user", "content": "Implement a queue."}]
        saved = copy.deepcopy(original)
        first = trial_messages(original, "code", 0)
        second = trial_messages(original, "code", 1)
        self.assertEqual([m["role"] for m in first], ["system", "user"])
        self.assertTrue(first[0]["content"].endswith("Write runnable code."))
        self.assertNotEqual(first[0]["content"], second[0]["content"])
        self.assertEqual(original, saved)

    def test_multimodal_message_is_copied_without_modification(self):
        original = [{"role": "user", "content": [{"type": "text", "text": "Read this."}]}]
        result = trial_messages(original, "vision", 0)
        self.assertEqual(result[1:], original)
        result[1]["content"][0]["text"] = "changed"
        self.assertEqual(original[0]["content"][0]["text"], "Read this.")


if __name__ == "__main__":
    unittest.main()
