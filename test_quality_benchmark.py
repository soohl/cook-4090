"""Checks for the deterministic experiment oracle and scoring distinctions."""
import json
import os
import re
import unittest
from unittest.mock import patch

import quality_benchmark as quality


class QualityHarnessTests(unittest.TestCase):
    def test_incremental_capacity_workloads_are_paired(self):
        with patch.dict(os.environ, QUALITY_OUTPUT="16384", QUALITY_INPUT_TOKENS="0"):
            a = list(quality.workloads(139264, 42, nominal_input=4096))
            b = list(quality.workloads(139264, 42, nominal_input=4096))
            compact = list(quality.workloads(139264, 42, nominal_input=0))
            self.assertEqual(a, b)
            self.assertEqual(a[1]["expected"], compact[1]["expected"])
            self.assertGreater(len(a[0]["messages"][-1]["content"][0]["text"]),
                               len(compact[0]["messages"][-1]["content"][0]["text"]))

    def test_ledger_oracle_from_rendered_records(self):
        with patch.dict(os.environ, QUALITY_OUTPUT="8192", QUALITY_INPUT_TOKENS="0"):
            for seed in (7, 42):
                cases = list(quality.workloads(32768, seed))
                ledger = cases[1]
                archive = ledger["messages"][-1]["content"][0]["text"]
                records = re.findall(r"LEDGER event (\d+): transfer (\d+) units from (account_\d) to (account_\d)", archive)
                self.assertEqual([int(r[0]) for r in records], list(range(1, 49)))
                incoming, outgoing = {}, {}
                for _, amount, source, destination in records:
                    outgoing[source] = outgoing.get(source, 0) + int(amount)
                    incoming[destination] = incoming.get(destination, 0) + int(amount)
                independent = {name: 1000 + incoming.get(name, 0) - outgoing.get(name, 0)
                               for name in ledger["expected"]["balances"]}
                self.assertEqual(independent, ledger["expected"]["balances"])
                self.assertEqual(sum(independent.values()), 8000)
                self.assertTrue(quality.score(json.dumps(ledger["expected"]), ledger["expected"])["exact"])

    def test_value_recall_does_not_mask_wrong_associations(self):
        expected = dict(codes={"a": "123", "b": "456"}, title="chart", red_circles=3, blue_position="left")
        answer = dict(expected, codes={"a": "456", "b": "123"})
        result = quality.score(json.dumps(answer), expected)
        self.assertFalse(result["exact"])
        self.assertEqual(result["correct_fields"], 3)
        self.assertEqual(result["code_value_recall"], 2)
        self.assertFalse(quality.score("", expected)["valid_json"])

    def test_balance_shape_separate_from_correct_values(self):
        expected = {"balances": {"account_0": 1100, "account_1": 900}, "total": 2000}
        flat = {"account_0": 1100, "account_1": 900, "total": 2000}
        result = quality.score(json.dumps(flat), expected)
        self.assertFalse(result["exact"])
        self.assertEqual(result["correct_balances"], 2)
        self.assertTrue(result["conserves_total"])


if __name__ == "__main__":
    unittest.main()
