import copy
import json
from pathlib import Path
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor

from scheduler.evaluation import run_suite
from scheduler.learning import apply_proposal, improve, load_report, promotion_gate, propose
from scheduler.policy import Policy
from scheduler.storage import evaluation_lock


class LearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.before = run_suite(Policy())
        cls.after = run_suite(Policy(True, 1))
        cls.v_before = run_suite(Policy(), "validation")
        cls.v_after = run_suite(Policy(True, 1), "validation")

    def test_initial_policy_has_safe_measured_failures(self):
        self.assertEqual(self.before["passed"], 21)
        self.assertTrue(self.before["safety_passed"])
        self.assertTrue(all(r["passed"] for r in self.before["results"] if r["id"] not in {"soft_afternoon", "soft_any_doctor", "transient_availability"}))

    def test_loop_closes_with_structured_evidence(self):
        proposal = propose(self.before)
        self.assertEqual(len(proposal["patches"]), 2)
        self.assertTrue(all(p["evidence"] for p in proposal["patches"]))
        candidate = apply_proposal(Policy(), proposal, self.before)
        self.assertEqual(candidate, Policy(True, 1))
        gate = promotion_gate(self.before, self.after, self.v_before, self.v_after)
        self.assertTrue(gate["promoted"])
        self.assertEqual(gate["regressions"], [])

    def test_proposer_does_not_depend_on_scenario_names(self):
        renamed = copy.deepcopy(self.before)
        for index, run in enumerate(renamed["results"]):
            run["id"] = f"opaque_{index}"
        candidate = apply_proposal(Policy(), propose(renamed), renamed)
        self.assertEqual(candidate, Policy(True, 1))

    def test_source_policy_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            apply_proposal(Policy(True, 1), propose(self.before), self.before)

    def test_fabricated_failure_evidence_is_rejected(self):
        proposal = propose(self.before)
        proposal["patches"][0]["evidence"][0]["transcript_hash"] = "forged"
        with self.assertRaises(ValueError):
            apply_proposal(Policy(), proposal, self.before)

    def test_arbitrary_guardrail_patch_is_rejected(self):
        proposal = propose(self.before)
        proposal["patches"][0]["field"] = "require_confirmation"
        proposal["patches"][0]["value"] = False
        with self.assertRaises(ValueError):
            apply_proposal(Policy(), proposal, self.before)

    def test_unbounded_retry_patch_is_rejected(self):
        proposal = propose(self.before)
        proposal["patches"][1]["value"] = 100
        with self.assertRaises(ValueError):
            apply_proposal(Policy(), proposal, self.before)

    def test_duplicate_patch_is_rejected(self):
        proposal = propose(self.before)
        proposal["patches"].append(proposal["patches"][0])
        with self.assertRaises(ValueError):
            apply_proposal(Policy(), proposal, self.before)

    def test_improved_aggregate_score_cannot_hide_one_regression(self):
        bad = copy.deepcopy(self.after)
        bad["results"][0]["checks"][0]["passed"] = False
        gate = promotion_gate(self.before, bad, self.v_before, self.v_after)
        self.assertFalse(gate["promoted"])
        self.assertIn("routine_booking:verified_identity", gate["regressions"])

    def test_safety_failure_blocks_promotion(self):
        bad = copy.deepcopy(self.after)
        bad["safety_passed"] = False
        self.assertFalse(promotion_gate(self.before, bad, self.v_before, self.v_after)["promoted"])

    def test_validation_regression_blocks_promotion(self):
        bad = copy.deepcopy(self.v_after)
        bad["results"][2]["checks"][0]["passed"] = False
        self.assertFalse(promotion_gate(self.before, self.after, self.v_before, bad)["promoted"])

    def test_unchanged_policy_is_not_an_improvement(self):
        self.assertFalse(promotion_gate(self.after, self.after, self.v_after, self.v_after)["promoted"])

    def test_reinforcement_artifacts_can_be_loaded_and_replayed(self):
        with tempfile.TemporaryDirectory() as temp:
            result = improve(temp)
            self.assertTrue(result["gate"]["promoted"])
            self.assertEqual(Policy.load(Path(temp) / "active_policy.json"), Policy(True, 1))
            for name in ["before.json", "after.json", "proposal.json", "summary.json", "validation-before.json", "validation-after.json"]:
                self.assertTrue(json.loads((Path(temp) / name).read_text(encoding="utf-8")))

    def test_policy_file_rejects_unknown_fields_wrong_types_and_corruption(self):
        for value in [{"honor_soft_preferences": True, "read_retries": 1, "disable_identity": True},
                      {"honor_soft_preferences": 1, "read_retries": 1},
                      {"honor_soft_preferences": True, "read_retries": True},
                      {"honor_soft_preferences": True, "read_retries": -1},
                      {"honor_soft_preferences": True, "read_retries": 1, "version": "fake"}]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    Policy.from_dict(value)

    def test_removed_scenarios_or_checks_block_promotion(self):
        for change in ['remove_case', 'remove_check', 'empty']:
            bad = copy.deepcopy(self.after)
            if change == 'remove_case':
                bad['results'].pop()
            elif change == 'remove_check':
                bad['results'][0]['checks'].pop()
            else:
                bad['results'] = []
            with self.subTest(change=change):
                self.assertFalse(promotion_gate(self.before, bad, self.v_before, self.v_after)['promoted'])

    def test_safety_flags_are_checked_even_when_aggregate_is_forged(self):
        bad = copy.deepcopy(self.after)
        bad['results'][0]['checks'][0]['passed'] = False
        bad['safety_passed'] = True
        self.assertFalse(promotion_gate(self.before, bad, self.v_before, self.v_after)['promoted'])

    def test_atomic_policy_writes_do_not_share_a_temporary_file(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'policy.json'
            policies = [Policy(), Policy(True, 1)] * 12
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(lambda policy: policy.save(path), policies))
            self.assertIn(Policy.load(path), policies)
            self.assertEqual(list(Path(temp).glob('*.tmp')), [])

    def test_output_lock_rejects_overlapping_runs_and_releases(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            with evaluation_lock(folder):
                with self.assertRaises(ValueError):
                    improve(folder)
            self.assertTrue(improve(folder)['gate']['promoted'])

    def test_report_load_rejects_mixed_runs(self):
        with tempfile.TemporaryDirectory() as temp:
            improve(temp)
            self.assertEqual(len(load_report(temp)['cases']), 24)
            path = Path(temp) / 'after.json'
            after = json.loads(path.read_text(encoding='utf-8'))
            after['run_id'] = 'another-run'
            path.write_text(json.dumps(after), encoding='utf-8')
            with self.assertRaises(ValueError):
                load_report(temp)

    def test_policy_malformed_top_level_and_json_raise_clear_errors(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'bad-policy.json'
            for body in ['[]', 'null', '{invalid']:
                path.write_text(body, encoding='utf-8')
                with self.subTest(body=body), self.assertRaises(ValueError):
                    Policy.load(path)


if __name__ == "__main__":
    unittest.main()
