from datetime import date, datetime, timedelta
import unittest
from unittest.mock import patch

from scheduler.agent import Agent
from scheduler.clinic import BookingUnavailable, Clinic, IST, TransientError
from scheduler.evaluation import judge, run_case, run_suite
from scheduler.parsing import Clarification, Request, parse_request
from scheduler.policy import Policy


class ConversationRegressionTests(unittest.TestCase):
    def setUp(self):
        self.clinic = Clinic(today=date(2026, 10, 7))
        self.agent = Agent(self.clinic, Policy(True, 1))

    def tearDown(self):
        self.clinic.close()

    def preview(self, agent=None):
        agent = agent or self.agent
        agent.reply('General tomorrow with Dr Rao')
        agent.reply('Name: Maya Chen, DOB: 1990-04-12')
        agent.reply('1')
        self.assertEqual(agent.stage, 'confirm')
        return agent

    def test_invalid_corrections_cannot_reuse_old_preview(self):
        corrections = [
            'Actually make that 16:00 PST', 'Change to 2026-02-30',
            '14:00 or 16:00', 'General with Dr Lin', 'Dr Jones tomorrow',
            'Tomorrow or Friday', 'At 16:00 CET', 'At 16:00 JST',
            'Not 14:00', 'Between 2 and 4pm', 'Change to October 9',
            'Cardiology tomorrow', 'Afternoon at 09:00', 'Actually make that later',
        ]
        for policy in [Policy(), Policy(True, 1)]:
            for text in corrections:
                with self.subTest(policy=policy, text=text):
                    agent = self.preview(Agent(self.clinic, policy))
                    agent.reply(text)
                    self.assertEqual(agent.stage, 'clarify')
                    self.assertIsNone(agent.selected)
                    self.assertIsNone(agent.tools._approved)
                    for reply in ['yes', '1', 'yes']:
                        agent.reply(reply)
                    self.assertEqual(agent.stage, 'clarify')
                    self.assertEqual(self.clinic.bookings(agent.session_id), [])

    def test_timezone_correction_requires_explicit_date_and_time_in_ist(self):
        self.preview()
        self.agent.reply('Actually 16:00 PST')
        self.agent.reply('16:00')
        self.assertEqual(self.agent.stage, 'clarify')
        self.agent.reply('2026-10-08 at 16:00 IST')
        self.assertEqual(self.agent.stage, 'choose')
        self.agent.reply('1')
        self.agent.reply('yes')
        self.assertEqual(self.agent.receipt['starts'][11:16], '16:00')

    def test_valid_date_resolves_invalid_date_without_reusing_confirmation(self):
        self.preview()
        self.agent.reply('2026-02-30')
        self.agent.reply('2026-10-09')
        self.assertEqual(self.agent.stage, 'choose')
        self.agent.reply('yes')
        self.assertEqual(self.clinic.bookings(), [])
        self.agent.reply('1')
        self.agent.reply('yes')
        self.assertEqual(self.agent.receipt['starts'][:10], '2026-10-09')

    def test_conflicting_doctor_and_service_requires_a_choice(self):
        agent = self.agent
        agent.reply('General tomorrow with Dr Lin')
        agent.reply('Name: Maya Chen, DOB: 1990-04-12')
        self.assertEqual(agent.stage, 'clarify')
        agent.reply('Dermatology with Dr Lin')
        self.assertEqual(agent.stage, 'request')
        agent.reply('tomorrow')
        agent.reply('1')
        agent.reply('yes')
        self.assertEqual(agent.receipt['doctor'], 'Dr Lin')

    def test_identity_change_after_verification_closes_conversation(self):
        self.preview()
        self.agent.reply('Name: Alex Rivera, DOB: 1985-09-23')
        self.agent.reply('yes')
        self.assertEqual(self.agent.stage, 'handoff')
        self.assertEqual(self.clinic.bookings(), [])

    def test_changed_canonical_slot_is_rejected_at_commit(self):
        self.preview()
        self.clinic.db.execute('UPDATE slots SET doctor=? WHERE id=?', ('Dr Patel', self.agent.selected['id']))
        response = self.agent.reply('yes')
        self.assertIn('changed', response)
        self.assertEqual(self.agent.stage, 'choose')
        self.assertEqual(self.clinic.bookings(), [])

    def test_commit_then_lost_result_recovers_existing_booking(self):
        self.preview()
        commit = self.agent.tools.book

        def lost_response():
            commit()
            raise BookingUnavailable('Lost result after commit')

        with patch.object(self.agent.tools, 'book', side_effect=lost_response):
            self.agent.reply('yes')
        self.assertEqual(self.agent.stage, 'booking_uncertain')
        self.assertEqual(len(self.clinic.bookings()), 1)
        self.agent.reply('retry')
        self.assertEqual(self.agent.stage, 'booked')
        self.assertEqual(len(self.clinic.bookings()), 1)
        self.assertIn(self.agent.receipt['id'], self.agent.history[-1]['text'])

    def test_failed_commit_checks_then_requires_new_confirmation(self):
        self.preview()
        with patch.object(self.agent.tools, 'book', side_effect=BookingUnavailable):
            self.agent.reply('yes')
        self.agent.reply('retry')
        self.assertEqual(self.agent.stage, 'confirm')
        self.assertEqual(self.clinic.bookings(), [])
        self.agent.reply('yes')
        self.assertEqual(len(self.clinic.bookings()), 1)

    def test_unavailable_status_read_never_repeats_mutation(self):
        self.preview()
        with patch.object(self.agent.tools, 'book', side_effect=BookingUnavailable):
            self.agent.reply('yes')
        with patch.object(self.agent.tools, 'find_booking', side_effect=TransientError):
            self.agent.reply('retry')
        self.assertEqual(self.agent.stage, 'booking_uncertain')
        self.assertEqual(self.clinic.bookings(), [])

    def test_unexpected_failure_preserves_paired_history_and_can_recover(self):
        self.preview()
        with patch.object(self.agent.tools, 'book', side_effect=RuntimeError):
            with self.assertRaises(RuntimeError):
                self.agent.reply('yes')
        self.assertEqual(len(self.agent.history), self.agent.turn * 2 + 1)
        self.assertEqual(self.agent.stage, 'booking_uncertain')
        self.agent.reply('retry')
        self.assertEqual(self.agent.stage, 'confirm')

    def test_conversation_limit_rejects_before_modifying_history(self):
        self.agent.turn = 120
        with self.assertRaises(ValueError):
            self.agent.reply('hello')
        self.assertEqual(len(self.agent.history), 1)

    def test_oracle_rejects_out_of_bounds_confirmation_without_crashing(self):
        self.preview()
        self.agent.reply('yes')
        self.clinic.db.execute('UPDATE bookings SET preview_turn=999')
        checks = judge(self.agent, {'bookings': 1})
        self.assertFalse(next(c for c in checks if c['name'] == 'explicit_exact_confirmation')['passed'])

    def test_oracle_requires_full_preview(self):
        self.preview()
        self.agent.reply('yes')
        self.agent.history[6]['text'] = 'Please confirm: your appointment'
        checks = judge(self.agent, {'bookings': 1})
        self.assertFalse(next(c for c in checks if c['name'] == 'explicit_exact_confirmation')['passed'])

    def test_evaluator_marks_runtime_failure_as_unsafe(self):
        scenario = {'id': 'runtime_fault', 'description': 'Tool raises unexpectedly',
                    'turns': ['Name: Maya Chen, DOB: 1990-04-12'], 'expect': {'bookings': 0}}
        with patch('scheduler.clinic.ToolGateway.verify_patient', side_effect=RuntimeError):
            result = run_case(scenario, Policy())
        self.assertFalse(result['passed'])
        self.assertIn('RUNTIME_ERROR', result['failure_codes'])
        self.assertFalse(next(c for c in result['checks'] if c['name'] == 'run_completed')['passed'])

    def test_empty_and_duplicate_suites_are_rejected(self):
        case = {'id': 'duplicate'}
        for cases in [[], [case, case]]:
            with self.assertRaises(ValueError):
                run_suite(Policy(), cases=cases)


class ClockAndParsingTests(unittest.TestCase):
    def test_supported_timezone_is_not_interpreted_as_a_clock(self):
        for zone in ['IST UTC+05:30', 'Asia/Kolkata', 'IST']:
            request, _ = parse_request(f'General tomorrow at 2pm {zone}', Request(), date(2026, 10, 7))
            self.assertEqual(request.exact_time, '14:00')
        request, _ = parse_request('General tomorrow IST UTC+05:30', Request(), date(2026, 10, 7))
        self.assertIsNone(request.exact_time)

    def test_ambiguous_or_unknown_fields_raise_clarification(self):
        for text in ['General tomorrow Dr Rao or Dr Patel', 'General tomorrow 08/10/2026',
                     'General tomorrow morning and afternoon', 'General tomorrow any time at 16:00',
                     'General tomorrow 2 to 4pm', 'General tomorrow at noon']:
            with self.subTest(text=text), self.assertRaises(Clarification):
                parse_request(text, Request(), date(2026, 10, 7))

    def test_same_day_search_excludes_started_appointments(self):
        current = datetime(2026, 10, 7, 13, 0, tzinfo=IST)
        clinic = Clinic(clock=lambda: current)
        try:
            agent = Agent(clinic)
            agent.reply('Name: Maya Chen, DOB: 1990-04-12')
            agent.reply('General today with Dr Rao')
            self.assertEqual([s['starts'][11:16] for s in agent.offers], ['14:00', '16:00'])
            agent.reply('1')
            current += timedelta(hours=2)
            agent.reply('yes')
            self.assertEqual(clinic.bookings(), [])
            self.assertEqual([s['starts'][11:16] for s in agent.offers], ['16:00'])
        finally:
            clinic.close()

    def test_day_rollover_extends_availability(self):
        current = datetime(2026, 10, 7, 23, 59, tzinfo=IST)
        clinic = Clinic(clock=lambda: current)
        try:
            agent = Agent(clinic)
            agent.reply('Name: Maya Chen, DOB: 1990-04-12')
            current += timedelta(minutes=2)
            agent.reply('General tomorrow with Dr Rao')
            self.assertEqual(agent.request.day, '2026-10-09')
            self.assertEqual(clinic.today, date(2026, 10, 8))
            agent.reply('General on 2026-10-22 with Dr Rao')
            self.assertEqual(agent.stage, 'choose')
        finally:
            clinic.close()

    def test_slot_that_expires_before_selection_is_not_previewed(self):
        current = datetime(2026, 10, 7, 8, 0, tzinfo=IST)
        clinic = Clinic(clock=lambda: current)
        try:
            agent = Agent(clinic)
            agent.reply('Name: Maya Chen, DOB: 1990-04-12')
            agent.reply('General today with Dr Rao')
            current += timedelta(hours=2)
            agent.reply('1')
            self.assertEqual(agent.stage, 'choose')
            self.assertIsNone(agent.selected)
            self.assertEqual(agent.offers[0]['starts'][11:16], '10:30')
        finally:
            clinic.close()
