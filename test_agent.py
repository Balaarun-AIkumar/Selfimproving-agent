from concurrent.futures import ThreadPoolExecutor
from datetime import date
import tempfile
from pathlib import Path
import unittest

from scheduler.agent import Agent
from scheduler.clinic import AccessDenied, Clinic, ToolGateway
from scheduler.evaluation import judge, load_scenarios, run_case
from scheduler.policy import Policy


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.clinic = Clinic(today=date(2026, 10, 7))
        self.agent = Agent(self.clinic, Policy(True, 1))

    def tearDown(self):
        self.clinic.close()

    def preview(self, agent=None):
        agent = agent or self.agent
        agent.reply("General tomorrow with Dr Rao")
        agent.reply("Name: Maya Chen, DOB: 1990-04-12")
        agent.reply("1")
        return agent

    def test_all_training_scenarios_after_reinforcement(self):
        for scenario in load_scenarios("training"):
            with self.subTest(scenario=scenario["id"]):
                run = run_case(scenario, Policy(True, 1))
                self.assertTrue(run["passed"], run["failure_codes"])

    def test_all_validation_scenarios_after_reinforcement(self):
        for scenario in load_scenarios("validation"):
            with self.subTest(scenario=scenario["id"]):
                run = run_case(scenario, Policy(True, 1))
                self.assertTrue(run["passed"], run["failure_codes"])

    def test_direct_booking_without_identity_is_denied(self):
        with self.assertRaises(AccessDenied):
            self.agent.tools.book()
        self.assertEqual(self.clinic.bookings(), [])

    def test_direct_search_without_identity_is_denied(self):
        with self.assertRaises(AccessDenied):
            self.agent.tools.search_slots(self.agent.request)

    def test_preview_cannot_be_confirmed_in_same_turn(self):
        self.preview()
        with self.assertRaises(AccessDenied):
            self.agent.tools.approve()

    def test_direct_booking_without_confirmation_is_denied(self):
        self.preview()
        with self.assertRaises(AccessDenied):
            self.agent.tools.book()

    def test_changed_date_does_not_reuse_yes(self):
        self.preview()
        self.agent.reply("yes, but change to 2026-10-09")
        self.assertEqual(self.agent.stage, "choose")
        self.assertEqual(self.clinic.bookings(), [])
        with self.assertRaises(AccessDenied):
            self.agent.tools.book()

    def test_changed_time_does_not_reuse_confirmation(self):
        self.preview()
        self.agent.reply("yes, but 16:00")
        self.assertEqual(self.agent.stage, "choose")
        self.assertEqual(self.clinic.bookings(), [])
        self.agent.reply("1")
        self.agent.reply("yes")
        self.assertEqual(self.clinic.bookings()[0]["starts"][11:16], "16:00")

    def test_simultaneous_patient_booking_is_atomic(self):
        other = Agent(self.clinic, Policy(True, 1))
        self.preview()
        self.preview(other)
        with ThreadPoolExecutor(max_workers=2) as pool:
            replies = list(pool.map(lambda agent: agent.reply("yes"), [self.agent, other]))
        self.assertEqual(len(self.clinic.bookings()), 1)
        self.assertEqual(sum("Booked. Reference" in r for r in replies), 1)
        self.assertEqual(sum("slot was just taken" in r for r in replies), 1)

    def test_database_unique_constraint_across_connections(self):
        with tempfile.TemporaryDirectory() as temp:
            a = Clinic(Path(temp) / "clinic.sqlite3", today=date(2026, 10, 7))
            b = Clinic(Path(temp) / "clinic.sqlite3", today=date(2026, 10, 7))
            try:
                agents = [self.preview(Agent(clinic, Policy(True, 1))) for clinic in [a, b]]
                with ThreadPoolExecutor(max_workers=2) as pool:
                    list(pool.map(lambda agent: agent.reply("yes"), agents))
                self.assertEqual(len(a.bookings()), 1)
            finally:
                a.close()
                b.close()

    def test_tool_level_booking_retry_is_idempotent(self):
        self.preview()
        self.agent.reply("yes")
        replay = self.agent.tools.book()
        self.assertEqual(replay["id"], self.agent.receipt["id"])
        self.assertEqual(len(self.clinic.bookings()), 1)

    def test_changed_slot_cannot_reuse_idempotency_key(self):
        self.preview()
        self.agent.reply("yes")
        key = self.agent.tools._approved["key"]
        slot = self.agent.offers[1]
        self.agent.tools.turn += 1
        self.agent.tools.preview(slot, key)
        self.agent.tools.turn += 1
        self.agent.tools.approve()
        with self.assertRaises(AccessDenied):
            self.agent.tools.book()
        self.assertEqual(len(self.clinic.bookings()), 1)

    def test_canonical_slot_fields_cannot_be_forged(self):
        self.preview()
        forged = {**self.agent.selected, "doctor": "Dr Fake"}
        with self.assertRaises(AccessDenied):
            self.agent.tools.preview(forged, "forged")

    def test_audit_oracle_detects_invented_receipt(self):
        self.agent.history.append({"role": "assistant", "text": "Booked. Reference CL-DEADBEEF"})
        checks = judge(self.agent, {"bookings": 0})
        self.assertFalse(next(c for c in checks if c["name"] == "grounded_receipts")["passed"])

    def test_oracle_checks_patient_confirmation_not_just_tool_audit(self):
        self.preview()
        self.agent.reply("yes")
        self.agent.history[-2]["text"] = "no"
        checks = judge(self.agent, {"bookings": 1})
        self.assertFalse(next(c for c in checks if c["name"] == "explicit_exact_confirmation")["passed"])

    def test_invalid_calendar_date_is_clarified(self):
        response = self.agent.reply("General on 2026-02-30")
        self.assertIn("date is invalid", response)
        self.assertEqual(self.clinic.bookings(), [])

    def test_unsupported_time_constraints_cannot_silently_book(self):
        for request in ["General tomorrow before 16:00", "General tomorrow, not afternoon", "General tomorrow at 25:61"]:
            with self.subTest(request=request):
                agent = Agent(self.clinic, Policy(True, 1))
                agent.reply("Name: Maya Chen, DOB: 1990-04-12")
                response = agent.reply(request)
                self.assertTrue("clarify" in response or "invalid" in response)
                agent.reply("yes")
                self.assertEqual(self.clinic.bookings(agent.session_id), [])

    def test_curly_apostrophe_urgent_message_stops_booking(self):
        self.preview()
        self.agent.reply("I can’t breathe")
        self.assertEqual(self.agent.stage, "escalated")
        self.assertEqual(self.clinic.bookings(), [])

    def test_identity_date_does_not_become_appointment_date(self):
        self.agent.reply("Name: Maya Chen, DOB: 1990-04-12")
        self.assertIsNone(self.agent.request.day)
        self.assertTrue(self.agent.tools.patient_id)

    def test_dob_is_redacted_from_export(self):
        self.agent.reply("Name: Maya Chen, DOB: 1990-04-12")
        self.assertNotIn("1990-04-12", str(self.agent.snapshot()))

    def test_empty_and_oversized_messages_rejected_before_turn(self):
        for message in ["", " ", None, "x" * 2001]:
            with self.subTest(message_type=type(message).__name__):
                with self.assertRaises(ValueError):
                    self.agent.reply(message)
        self.assertEqual(self.agent.turn, 0)

    def test_policy_is_pinned_for_existing_conversation(self):
        self.assertEqual(self.agent.policy, Policy(True, 1))
        self.assertEqual(Agent(self.clinic, Policy()).policy, Policy())


if __name__ == "__main__":
    unittest.main()
