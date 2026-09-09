import unittest
from vision_quality_benchmark import score


class VisionScoreTests(unittest.TestCase):
    def test_exact_match_requires_nested_json_types(self):
        for value in ('true', '1.0', '"1"'):
            with self.subTest(value=value):
                result = score('{"rows":[{"count":' + value + '}]}',
                               {"rows": [{"count": 1}]})
                self.assertFalse(result["exact"])
                self.assertEqual(result["correct"], 0)
        self.assertTrue(score('{"rows":[{"count":1}]}',
                              {"rows": [{"count": 1}]})["exact"])
        self.assertFalse(score('{"rows":{}}', {"rows": []})["exact"])

    def test_transcription_types_order_and_partial_credit(self):
        expected = {"rows": [{"id": "AB12", "quantity": 4, "price": "2.00"},
                             {"id": "CD34", "quantity": 7, "price": "3.10"}]}
        self.assertEqual(score('not json', expected)["correct"], 0)
        actual = '{"rows":[{"id":"AB12","quantity":4,"price":2},{"id":"CD34","quantity":7,"price":"3.10"}]}'
        result = score(actual, expected)
        self.assertFalse(result["exact"])
        self.assertEqual((result["correct"], result["total"]), (5, 6))

    def test_correct_values_with_extra_key_are_not_exact(self):
        result = score('{"Aster":20,"unexpected":0}', {"Aster": 20})
        self.assertFalse(result["exact"])
        self.assertEqual(result["correct"], 1)


if __name__ == "__main__":
    unittest.main()
