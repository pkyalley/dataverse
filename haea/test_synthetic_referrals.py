import unittest
from synthetic_referrals import generate, evaluate

class SyntheticExperimentTests(unittest.TestCase):
    def test_reproducible(self):
        self.assertEqual(list(generate(50, 42)), list(generate(50, 42)))
    def test_unique_identifiers(self):
        rows = list(generate(100))
        self.assertEqual(len(set(r["referral_id"] for r in rows)), 100)
    def test_rule_detects_defined_defects(self):
        metrics = evaluate(generate(1000))
        self.assertEqual(metrics["fp"], 0)
        self.assertEqual(metrics["fn"], 0)
    def test_zero_defects(self):
        m = evaluate(generate(50, missing_probability=0, invalid_date_probability=0))
        self.assertEqual(m["tp"], 0)
        self.assertEqual(m["tn"], 50)
    def test_invalid_probability(self):
        with self.assertRaises(ValueError):
            list(generate(1, missing_probability=1.5))

if __name__ == "__main__":
    unittest.main()
