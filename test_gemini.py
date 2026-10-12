import copy
from datetime import date
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from scheduler.agent import Agent
from scheduler.clinic import Clinic
from scheduler.gemini import GeminiClient, GeminiError, Settings
from scheduler.gemini_learning import apply_generated, suite_cases, improve_gemini
from scheduler.language import GeminiInterpreter, local_control
from scheduler.policy import Policy


class FakeClient:
    model = 'test-model'
    def __init__(self, value):
        self.value = value
        self.inputs = []
    def generate(self, instruction, data, schema, **kwargs):
        self.inputs.append(data)
        if isinstance(self.value, Exception):
            raise self.value
        return copy.deepcopy(self.value)


class GeminiTests(unittest.TestCase):
    def setUp(self):
        self.clinic = Clinic(today=date(2026, 10, 7))
        self.addCleanup(self.clinic.close)

    def ready_agent(self, decision):
        agent = Agent(self.clinic, Policy(True, 1))
        for message in ['General tomorrow with Dr Rao', 'Name: Maya Chen, DOB: 1990-04-12', '1']:
            agent.reply(message)
        agent.interpreter = GeminiInterpreter(FakeClient(decision))
        return agent

    def test_conversational_request_uses_model_then_real_tools(self):
        client = FakeClient({'action':'update','canonical_request':'general Dr Rao 2026-10-08','choice':None})
        agent = Agent(self.clinic, interpreter=GeminiInterpreter(client))
        agent.reply('A physical with Rao on the day after today')
        agent.reply('Name: Maya Chen, DOB: 1990-04-12')
        agent.reply('1')
        agent.reply('yes')
        self.assertEqual(agent.stage, 'booked')
        self.assertEqual(len(client.inputs), 1)
        self.assertEqual(agent.receipt['starts'][:10], '2026-10-08')

    def test_model_has_no_confirm_action(self):
        agent = self.ready_agent({'action':'confirm','canonical_request':'','choice':None})
        agent.reply('Maybe, I am not sure')
        agent.reply('yes')
        self.assertEqual(self.clinic.bookings(agent.session_id), [])
        self.assertTrue(agent.model_errors)

    def test_model_cannot_forge_an_option_or_mix_a_correction(self):
        for decision in [{'action':'choose','canonical_request':'','choice':99},
                         {'action':'choose','canonical_request':'general tomorrow','choice':1}]:
            agent = Agent(self.clinic, interpreter=GeminiInterpreter(FakeClient(decision)))
            agent.reply('Pick an appointment for me')
            self.assertTrue(agent.model_errors)
            self.assertFalse(agent.receipt)

    def test_declined_preview_cannot_be_committed_by_later_yes(self):
        agent = self.ready_agent({'action':'decline','canonical_request':'','choice':None})
        agent.reply("That slot won't work for me")
        agent.reply('yes')
        self.assertEqual(self.clinic.bookings(agent.session_id), [])

    def test_correction_containing_yes_never_commits_old_preview(self):
        agent = self.ready_agent({'action':'update','canonical_request':'2026-10-09','choice':None})
        agent.reply('Yes but Friday would be better')
        agent.reply('yes')
        self.assertEqual(self.clinic.bookings(agent.session_id), [])
        self.assertEqual(agent.request.day, '2026-10-09')

    def test_outage_invalidates_pending_preview(self):
        agent = self.ready_agent(GeminiError('Gemini unavailable'))
        agent.reply('Actually, a later appointment')
        agent.reply('yes')
        self.assertEqual(self.clinic.bookings(agent.session_id), [])
        self.assertIn('Gemini unavailable', agent.model_errors[0]['error'])

    def test_generated_unknown_doctor_rejected_by_local_parser(self):
        agent = Agent(self.clinic, interpreter=GeminiInterpreter(FakeClient(
            {'action':'update','canonical_request':'general Dr Invented 2026-10-08','choice':None})))
        self.assertIn('not in this demo clinic', agent.reply('Can you find a doctor?'))
        self.assertFalse(agent.request.doctor)

    def test_failed_correction_invalidates_old_offers(self):
        agent = Agent(self.clinic)
        agent.reply('General tomorrow with Dr Rao')
        agent.reply('Name: Maya Chen, DOB: 1990-04-12')
        self.assertEqual(agent.stage, 'choose')
        agent.interpreter = GeminiInterpreter(FakeClient(GeminiError('Quota reached')))
        agent.reply('Actually, Friday afternoon instead')
        agent.reply('1')
        agent.reply('yes')
        self.assertFalse(agent.offers)
        self.assertEqual(self.clinic.bookings(agent.session_id), [])

    def test_birth_dates_are_not_sent_to_model(self):
        client = FakeClient({'action':'none','canonical_request':'','choice':None})
        agent = Agent(self.clinic, interpreter=GeminiInterpreter(client))
        agent.reply('Born 1990-04-12, and I need some help scheduling')
        self.assertNotIn('1990-04-12', json.dumps(client.inputs))

    def test_exact_confirmation_stays_local(self):
        agent = self.ready_agent(GeminiError('Model must not be called'))
        agent.reply('yes')
        self.assertEqual(agent.stage, 'booked')
        self.assertFalse(agent.model_errors)

    def test_ambiguous_reply_keeps_preview_without_booking(self):
        from scheduler.evaluation import judge
        agent = self.ready_agent({'action':'none','canonical_request':'','choice':None})
        selected = copy.deepcopy(agent.selected)
        tools = copy.deepcopy(agent.tools.events)
        for message in ['sounds interesting', 'Let me think.', 'That might work.']:
            reply = agent.reply(message)
            self.assertEqual(agent.stage, 'confirm')
            self.assertEqual(agent.selected, selected)
            self.assertEqual(agent.tools.events, tools)
            self.assertIn('Nothing is booked yet', reply)
            self.assertFalse(self.clinic.bookings(agent.session_id))
        agent.reply('yes')
        self.assertEqual(agent.stage, 'booked')
        self.assertEqual(len(self.clinic.bookings(agent.session_id)), 1)
        self.assertTrue(all(c['passed'] for c in judge(agent, {'bookings':1})))

    def test_unrecognized_model_update_still_parses_original_correction(self):
        agent = self.ready_agent({'action':'none','canonical_request':'','choice':None})
        agent.reply('Actually, change it to Friday')
        agent.reply('yes')
        self.assertEqual(agent.request.day, '2026-10-09')
        self.assertIsNone(agent.selected)
        self.assertFalse(self.clinic.bookings(agent.session_id))

    def test_unrecognized_unsupported_time_still_invalidates_preview(self):
        agent = self.ready_agent({'action':'none','canonical_request':'','choice':None})
        agent.reply('Actually, after 15:00 instead')
        agent.reply('yes')
        self.assertEqual(agent.stage, 'clarify')
        self.assertIsNone(agent.selected)
        self.assertFalse(self.clinic.bookings(agent.session_id))

    def test_control_recognizer_does_not_hide_mixed_corrections(self):
        self.assertTrue(local_control('yes'))
        self.assertTrue(local_control('Name: Maya Chen, DOB: 1990-04-12'))
        self.assertFalse(local_control('yes, but change it to Friday'))

    def evidence(self):
        return {'policy':Policy().to_dict(),'results':[{'id':'opaque-failure','passed':False,
                'transcript_hash':'evidence-hash','failure_codes':['NEW_LANGUAGE_FAILURE']}]}

    def proposal(self, field='interpretation_notes', value='Resolve relative week offsets using the supplied clinic date.'):
        return {'analysis':'The observed language was misunderstood.', 'patches':[{
            'field':field,'value_json':json.dumps(value),'reason':'The recorded conversation lost a supplied date.',
            'evidence_run_ids':['opaque-failure']}]}

    def test_new_prompt_repair_does_not_need_a_predefined_failure_code(self):
        candidate, proposal = apply_generated(Policy(), self.proposal(), self.evidence())
        self.assertIn('week offsets', candidate.interpretation_notes)
        self.assertEqual(proposal['patches'][0]['evidence'][0]['transcript_hash'], 'evidence-hash')
        self.assertNotEqual(candidate.version, Policy().version)

    def test_generated_safety_or_unbounded_changes_are_rejected(self):
        for field, value in [('require_confirmation',False),('read_retries',99),('read_retries',True),
                             ('interpretation_notes','x'*4001),('honor_soft_preferences','true')]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                apply_generated(Policy(), self.proposal(field,value), self.evidence())

    def test_fabricated_or_passing_evidence_is_rejected(self):
        evidence = self.evidence()
        for change in ['passing','missing']:
            bad = copy.deepcopy(evidence)
            if change == 'passing': bad['results'][0]['passed'] = True
            else: bad['results'][0]['id'] = 'different'
            with self.assertRaises(ValueError): apply_generated(Policy(), self.proposal(), bad)

    def test_foreign_policy_evidence_is_rejected(self):
        with self.assertRaises(ValueError):
            apply_generated(Policy(True,1), self.proposal(), self.evidence())

    def test_prompt_policy_round_trip_and_old_version_compatibility(self):
        self.assertEqual(Policy().version, '5c89fe83b4fb')
        candidate = Policy(True,1,'Interpret a fortnight as fourteen days.')
        self.assertEqual(Policy.from_dict(candidate.to_dict()), candidate)

    def test_train_and_validation_names_do_not_overlap(self):
        train, validation = suite_cases('training'), suite_cases('validation')
        self.assertEqual(len(train),32)
        self.assertEqual(len(validation),12)
        self.assertFalse({c['id'] for c in train} & {c['id'] for c in validation})

    def test_settings_repr_does_not_contain_key(self):
        settings = Settings('secret-for-test', interval=0)
        self.assertNotIn('secret-for-test', repr(settings))

    def test_request_cache_and_budget(self):
        client = GeminiClient(Settings('secret-for-test', max_calls=1, interval=0))
        payload = json.dumps({'candidates':[{'finishReason':'STOP','content':{'parts':[{'text':'{"ok":true}'}]}}]}).encode()
        with patch.object(client._opener, 'open', return_value=io.BytesIO(payload)):
            self.assertEqual(client.generate('test',{},{}), {'ok':True})
        self.assertEqual(client.generate('test',{},{}), {'ok':True})
        self.assertEqual(client.stats()['cache_hits'],1)
        with self.assertRaisesRegex(GeminiError,'budget'):
            client.generate('different',{}, {})

    def test_provider_error_does_not_echo_key_or_remote_body(self):
        client = GeminiClient(Settings('secret-for-test',interval=0))
        failure = HTTPError('https://provider',403,'secret-for-test',{},None)
        with patch.object(client._opener,'open',side_effect=failure):
            with self.assertRaises(GeminiError) as caught: client.generate('test',{}, {})
        self.assertNotIn('secret-for-test',str(caught.exception))
        self.assertIn('permit',str(caught.exception))

    def test_evaluation_cache_survives_restart_without_saving_credentials(self):
        payload = json.dumps({'candidates':[{'finishReason':'STOP','content':{'parts':[{'text':'{"ok":true}'}]}}]}).encode()
        with tempfile.TemporaryDirectory() as folder:
            cache = Path(folder) / 'model-cache.json'
            settings = Settings('private-test-key', interval=0)
            first = GeminiClient(settings, cache_path=cache)
            with patch.object(first._opener, 'open', return_value=io.BytesIO(payload)):
                first.generate('test', {}, {})
            self.assertNotIn('private-test-key', cache.read_text())
            restarted = GeminiClient(settings, cache_path=cache)
            with patch.object(restarted._opener, 'open', side_effect=AssertionError('Unexpected network request')):
                self.assertEqual(restarted.generate('test', {}, {}), {'ok': True})
            self.assertEqual(restarted.stats()['requests'], 0)
            self.assertEqual(restarted.stats()['cache_hits'], 1)
            self.assertFalse(GeminiClient(settings, cache_path=cache, fresh=True)._cache)
            changed_model = Settings('private-test-key', model='different-model', interval=0)
            self.assertFalse(GeminiClient(changed_model, cache_path=cache)._cache)

    def test_corrupt_cache_is_ignored(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = Path(folder) / 'model-cache.json'
            cache.write_text('{broken', encoding='utf-8')
            self.assertFalse(GeminiClient(Settings('test-key'), cache_path=cache)._cache)

    def test_rate_limit_stops_without_automatic_paid_fallback(self):
        client = GeminiClient(Settings('secret-for-test',interval=0))
        with patch.object(client._opener,'open',side_effect=HTTPError('',429,'quota',{},None)):
            with self.assertRaisesRegex(GeminiError,'quota'): client.generate('test',{}, {})
        self.assertEqual(client.calls,1)

    def test_transient_generation_errors_have_a_bounded_retry(self):
        client = GeminiClient(Settings('secret-for-test',interval=0))
        with patch.object(client._opener,'open',side_effect=HTTPError('',503,'busy',{},None)), patch('scheduler.gemini.time.sleep'):
            with self.assertRaisesRegex(GeminiError,'503'): client.generate('test',{}, {})
        self.assertEqual(client.calls,3)

    def test_last_allowed_request_preserves_overload_error(self):
        client = GeminiClient(Settings('private-test-key', max_calls=1, interval=0))
        with patch.object(client._opener, 'open', side_effect=HTTPError('', 503, 'busy', {}, None)):
            with self.assertRaisesRegex(GeminiError, '503'):
                client.generate('test', {}, {})
        self.assertEqual(client.calls, 1)

    def test_repair_model_is_explicit_and_keeps_credentials_server_side(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict('os.environ', {}, clear=True):
            path = Path(folder) / '.env'
            path.write_text('GEMINI_API_KEY=private-test-key\nGEMINI_MODEL=primary\nGEMINI_REPAIR_MODEL=repair\n')
            settings = Settings.load(path)
            client = GeminiClient(settings)
            repair = client.repair_client()
            self.assertEqual(client.model, 'primary')
            self.assertEqual(repair.model, 'repair')
            self.assertNotIn('private-test-key', repr(repair.settings))
            path.write_text('GEMINI_API_KEY=private-test-key\nGEMINI_REPAIR_MODEL=https://wrong\n')
            with self.assertRaisesRegex(GeminiError, 'REPAIR_MODEL'):
                Settings.load(path)

    def test_interrupted_candidate_and_frozen_validation_resume_without_regeneration(self):
        from scheduler.evaluation import run_suite
        for interrupted_stage in ['candidate', 'validation']:
            with self.subTest(stage=interrupted_stage), tempfile.TemporaryDirectory() as folder:
                events = []
                class Proposer:
                    model = 'repair-model'
                    calls = 0
                    def stats(inner):
                        return {'requests': inner.calls}
                    def generate(inner, instruction, data, schema, **kwargs):
                        inner.calls += 1
                        evidence = [r['id'] for r in data['failed_runs']]
                        return {'analysis': 'Fix observed failures.', 'patches': [
                            {'field': name, 'value_json': json.dumps(value), 'reason': 'Observed failure.',
                             'evidence_run_ids': evidence}
                            for name, value in [('honor_soft_preferences', True), ('read_retries', 1)]]}
                proposer = Proposer()
                def interrupted(policy, split='training', cases=None, interpreter=None, progress=None):
                    events.append(split)
                    if (interrupted_stage == 'candidate' and len(events) == 2 or
                            interrupted_stage == 'validation' and len(events) == 4):
                        raise GeminiError('Temporary overload')
                    return run_suite(policy, split)
                with patch('scheduler.gemini_learning.run_suite', side_effect=interrupted):
                    with self.assertRaisesRegex(GeminiError, 'overload'):
                        improve_gemini(folder, FakeClient({}), proposer_client=proposer)
                checkpoint = json.loads((Path(folder) / 'checkpoint.json').read_text())['state']
                self.assertEqual(proposer.calls, 1)
                self.assertFalse((Path(folder) / 'active_policy.json').exists())
                if interrupted_stage == 'validation':
                    self.assertTrue(checkpoint['frozen'])
                    self.assertIn('validation_before', checkpoint)
                else:
                    self.assertIn('pending', checkpoint)
                resumed_events = []
                def resumed(policy, split='training', cases=None, interpreter=None, progress=None):
                    resumed_events.append(split)
                    return run_suite(policy, split)
                # A restarted process gets fresh clients; saved evidence and the proposal remain.
                with patch('scheduler.gemini_learning.run_suite', side_effect=resumed):
                    result = improve_gemini(folder, FakeClient({}), proposer_client=Proposer())
                self.assertTrue(result['gate']['promoted'])
                self.assertEqual(result['run_id'], checkpoint['run_id'])
                self.assertEqual(result['proposal_usage']['requests'], 1)
                self.assertTrue(result['resumed'])
                self.assertEqual(resumed_events.count('training'), 3 if interrupted_stage == 'candidate' else 2)

    def test_checkpoint_rejects_changed_experiment_and_damaged_contents(self):
        from scheduler.gemini_learning import experiment_key, load_checkpoint, save_checkpoint
        client = FakeClient({})
        key = experiment_key(Policy(), client, client, 2)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'checkpoint.json'
            state = {'run_id': 'saved-run', 'attempts': [], 'frozen': True}
            save_checkpoint(path, key, state)
            self.assertEqual(load_checkpoint(path, key), state)
            self.assertFalse(load_checkpoint(path, experiment_key(Policy(True, 1), client, client, 2)))
            changed = FakeClient({})
            changed.model = 'different-model'
            self.assertFalse(load_checkpoint(path, experiment_key(Policy(), client, changed, 2)))
            saved = json.loads(path.read_text())
            saved['state']['frozen'] = False
            path.write_text(json.dumps(saved))
            self.assertFalse(load_checkpoint(path, key))

    def test_fresh_experiment_bypasses_a_completed_checkpoint(self):
        from scheduler.evaluation import run_suite
        def replay(policy, split='training', *args, **kwargs):
            return run_suite(policy, split)
        # A passing baseline needs no proposal and isolates checkpoint reuse.
        with tempfile.TemporaryDirectory() as folder, patch('scheduler.gemini_learning.run_suite', side_effect=replay) as replayed:
            first = improve_gemini(folder, FakeClient({}), Policy(True, 1))
            resumed = improve_gemini(folder, FakeClient({}), Policy(True, 1))
            self.assertEqual(resumed['run_id'], first['run_id'])
            self.assertEqual(replayed.call_count, 11)  # Three measured stages once, plus four protected suites per run.
            fresh = improve_gemini(folder, FakeClient({}), Policy(True, 1), fresh=True)
            self.assertNotEqual(fresh['run_id'], first['run_id'])
            self.assertFalse(fresh['resumed'])

    def test_generated_candidate_replays_and_freezes_before_validation(self):
        from scheduler.evaluation import run_suite
        events = []
        measured = []
        class Proposer:
            model = 'mock-provider'
            def stats(self):
                return {'requests': 1}
            def generate(inner, instruction, data, schema, **kwargs):
                events.append('proposal')
                self.assertTrue(data['failed_runs'])
                evidence = [r['id'] for r in data['failed_runs']]
                self.assertFalse(set(evidence) & {r['id'] for r in suite_cases('validation')})
                return {'analysis': 'Repair observed preferences and transient reads.', 'patches': [
                    {'field': name, 'value_json': json.dumps(value), 'reason': 'Observed training failure.',
                     'evidence_run_ids': evidence}
                    for name, value in [('honor_soft_preferences', True), ('read_retries', 1),
                                        ('interpretation_notes', 'Retain preferences across clarification turns.')]]}
        def replay(policy, split='training', cases=None, interpreter=None, progress=None):
            events.append(split)
            # Real tool/database evaluation; only provider interpretation is bypassed in this test.
            return run_suite(policy, split)
        with tempfile.TemporaryDirectory() as folder, patch('scheduler.gemini_learning.run_suite', side_effect=replay):
            result = improve_gemini(folder, Proposer(), on_event=measured.append)
            self.assertTrue(result['gate']['promoted'])
            self.assertEqual(Policy.load(Path(folder) / 'active_policy.json').read_retries, 1)
            self.assertTrue((Path(folder) / 'report.md').exists())
            self.assertEqual(result['events'], measured)
            self.assertEqual([e['stage'] for e in measured], ['baseline', 'proposal', 'replay', 'validation', 'gate'])
            self.assertEqual(measured[0]['passed'], result['before']['passed'])
            self.assertTrue(measured[0]['failures'])
            self.assertEqual(measured[2]['passed'], result['after']['passed'])
            self.assertTrue(measured[-1]['promoted'])
        self.assertLess(events.index('proposal'), events.index('validation'))
        self.assertEqual(events.count('proposal'), 1)

    def test_interrupted_experiment_does_not_replace_saved_policy(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'active_policy.json'
            original = Policy(True, 1, 'Existing accepted guidance.')
            original.save(path)
            with patch('scheduler.gemini_learning.run_suite', side_effect=GeminiError('Quota reached')):
                with self.assertRaisesRegex(GeminiError, 'Quota'):
                    improve_gemini(folder, FakeClient({}), original)
            self.assertEqual(Policy.load(path), original)
            self.assertFalse((Path(folder) / 'summary.json').exists())

    def test_concurrent_generation_returns_busy_without_queueing(self):
        client = GeminiClient(Settings('test-key', interval=0))
        client._lock.acquire()
        try:
            with self.assertRaisesRegex(GeminiError, 'another request'):
                client.generate('test', {}, {})
            self.assertEqual(client.calls, 0)
        finally:
            client._lock.release()

    def test_malformed_provider_envelopes_are_recoverable(self):
        for envelope in [[], {'candidates': [None]}, {'candidates': [{'finishReason': 'STOP', 'content': None}]}]:
            client = GeminiClient(Settings('test-key', interval=0))
            with self.subTest(envelope=envelope), patch.object(client, '_request', return_value=json.dumps(envelope).encode()):
                with self.assertRaisesRegex(GeminiError, 'invalid structured data'):
                    client.generate('test', {}, {})
            self.assertFalse(client._lock.locked())

    def test_bad_usage_metadata_does_not_discard_valid_model_response(self):
        for usage in [None, [], {'totalTokenCount': 'invalid'}, {'totalTokenCount': -1}]:
            client = GeminiClient(Settings('test-key', interval=0))
            envelope = {'candidates': [{'finishReason': 'STOP', 'content': {'parts': [{'text': '{"ok":true}'}]}}], 'usageMetadata': usage}
            with self.subTest(usage=usage), patch.object(client, '_request', return_value=json.dumps(envelope).encode()):
                self.assertEqual(client.generate('test', {}, {}), {'ok': True})
            self.assertEqual(client.tokens, 0)

    def test_report_write_failure_does_not_replace_active_policy(self):
        from scheduler.evaluation import run_suite
        def replay(policy, split='training', **kwargs):
            return run_suite(policy, split)
        client = GeminiClient(Settings('test-key', interval=0))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'active_policy.json'
            Policy().save(path)
            with patch('scheduler.gemini_learning.run_suite', side_effect=lambda policy, split='training', *args, **kwargs: replay(policy, split)), patch('scheduler.gemini_learning.write_text', side_effect=OSError('Disk full')):
                with self.assertRaisesRegex(OSError, 'Disk full'):
                    improve_gemini(folder, client, Policy(True, 1))
            self.assertEqual(Policy.load(path), Policy())


if __name__ == '__main__':
    unittest.main()
