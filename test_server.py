import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from scheduler.server import DemoServer
from scheduler.policy import Policy
from scheduler.gemini import GeminiClient, GeminiError, Settings


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.server = DemoServer(("127.0.0.1", 0), self.temp.name)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()
        self.temp.cleanup()

    def call(self, path, body=None, headers=None):
        req = Request(self.base + path, data=json.dumps(body).encode() if body is not None else None,
                      headers={"Content-Type": "application/json", **(headers or {})})
        with urlopen(req, timeout=10) as response:
            return response.status, json.load(response)

    def test_http_conversation_commits_one_booking(self):
        _, session = self.call("/api/session", {})
        headers = {"X-Session-ID": session["session_id"]}
        for text in ["General next Monday with Dr Patel at 16:00", "Name: Maya Chen, DOB: 1990-04-12", "1", "yes", "yes"]:
            _, session = self.call("/api/chat", {"message": text}, headers)
        self.assertEqual(session["stage"], "booked")
        self.assertTrue(session["receipt"]["id"].startswith("CL-"))
        self.assertEqual(len(self.server.clinic.bookings(session["session_id"])), 1)

    def test_second_server_cannot_take_the_running_app_port(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(OSError):
                duplicate = DemoServer(self.server.server_address, folder)
                duplicate.server_close()
        _, health = self.call('/api/health')
        self.assertEqual(health['status'], 'ok')

    def test_missing_session_cannot_read_or_mutate_patient_state(self):
        with self.assertRaises(HTTPError) as exc:
            self.call("/api/session")
        self.assertEqual(exc.exception.code, 401)
        with self.assertRaises(HTTPError) as exc:
            self.call("/api/chat", {"message": "yes"})
        self.assertEqual(exc.exception.code, 401)

    def test_external_origin_cannot_create_session(self):
        with self.assertRaises(HTTPError) as exc:
            self.call("/api/session", {}, {"Origin": "https://unrelated.example"})
        self.assertEqual(exc.exception.code, 403)

    def test_cross_site_local_request_is_denied(self):
        with self.assertRaises(HTTPError) as exc:
            self.call("/api/session", {}, {"Sec-Fetch-Site": "cross-site"})
        self.assertEqual(exc.exception.code, 403)

    def test_health_and_static_app_have_no_external_dependencies(self):
        _, health = self.call("/api/health")
        self.assertTrue(health["synthetic_only"])
        with urlopen(self.base + "/") as response:
            html = response.read().decode()
            self.assertIn("CareLoop", html)
            self.assertIn("frame-ancestors 'none'", response.headers["Content-Security-Policy"])
        with urlopen(self.base + "/style.css") as response:
            self.assertNotIn("@import", response.read().decode())

    def test_evaluation_promotes_only_for_new_sessions(self):
        _, old = self.call("/api/session", {})
        self.assertFalse(old["policy"]["honor_soft_preferences"])
        _, loop = self.call("/api/improve", {})
        self.assertTrue(loop["gate"]["promoted"])
        _, old_state = self.call("/api/session", headers={"X-Session-ID":old["session_id"]})
        self.assertFalse(old_state["policy"]["honor_soft_preferences"])
        _, new = self.call("/api/session", {})
        self.assertTrue(new["policy"]["honor_soft_preferences"])

    def test_message_replay_returns_latest_state_without_another_turn(self):
        _, session = self.call('/api/session', {})
        headers = {'X-Session-ID': session['session_id']}
        body = {'message': 'General next Monday', 'message_id': 'first-request', 'expected_turn': 0}
        _, first = self.call('/api/chat', body, headers)
        _, replay = self.call('/api/chat', body, headers)
        self.assertEqual(first, replay)
        self.assertEqual(replay['turn'], 1)
        self.assertEqual(replay['acknowledged_message_ids'], ['first-request'])
        with self.assertRaises(HTTPError) as error:
            self.call('/api/chat', {**body, 'message': 'yes'}, headers)
        self.assertEqual(error.exception.code, 409)

    def test_stale_tab_cannot_confirm_changed_conversation(self):
        _, session = self.call('/api/session', {})
        headers = {'X-Session-ID': session['session_id']}
        self.call('/api/chat', {'message': 'General next Monday', 'expected_turn': 0}, headers)
        with self.assertRaises(HTTPError) as error:
            self.call('/api/chat', {'message': 'yes', 'expected_turn': 0}, headers)
        self.assertEqual(error.exception.code, 409)
        self.assertEqual(self.server.sessions[session['session_id']]['agent'].turn, 1)
        self.assertEqual(self.server.clinic.bookings(), [])

    def test_replaying_a_confirmation_returns_one_receipt(self):
        _, session = self.call('/api/session', {})
        headers = {'X-Session-ID': session['session_id']}
        for text in ['General next Monday with Dr Rao', 'Name: Maya Chen, DOB: 1990-04-12', '1']:
            _, session = self.call('/api/chat', {'message': text}, headers)
        confirmation = {'message': 'yes', 'message_id': 'confirm-once', 'expected_turn': session['turn']}
        _, first = self.call('/api/chat', confirmation, headers)
        _, replay = self.call('/api/chat', confirmation, headers)
        self.assertEqual(replay['receipt'], first['receipt'])
        self.assertEqual(replay['turn'], first['turn'])
        self.assertEqual(len(self.server.clinic.bookings()), 1)

    def test_session_capacity_is_bounded_and_expired_capacity_is_reused(self):
        self.server.sessions = {str(i): {'last_used': time.monotonic()} for i in range(128)}
        with self.assertRaises(HTTPError) as error:
            self.call('/api/session', {})
        self.assertEqual(error.exception.code, 429)
        self.server.sessions['0']['last_used'] -= 1801
        self.call('/api/session', {})
        self.assertEqual(len(self.server.sessions), 128)
        self.assertNotIn('0', self.server.sessions)

    def test_worker_capacity_returns_service_unavailable_and_recovers(self):
        for _ in range(24):
            self.server.workers.acquire()
        try:
            with self.assertRaises(HTTPError) as error:
                self.call('/api/health')
            self.assertEqual(error.exception.code, 503)
        finally:
            for _ in range(24):
                self.server.workers.release()
        self.assertEqual(self.call('/api/health')[0], 200)

    def test_expired_session_requires_new_conversation(self):
        _, session = self.call('/api/session', {})
        self.server.sessions[session['session_id']]['last_used'] = time.monotonic() - 1801
        with self.assertRaises(HTTPError) as error:
            self.call('/api/session', headers={'X-Session-ID': session['session_id']})
        self.assertEqual(error.exception.code, 401)
        self.assertNotIn(session['session_id'], self.server.sessions)

    def test_initial_policy_comparison_remains_available_after_promotion(self):
        self.call('/api/improve', {})
        _, session = self.call('/api/session', {'baseline': True})
        self.assertFalse(session['policy']['honor_soft_preferences'])
        self.assertEqual(session['policy']['read_retries'], 0)

    def test_saved_report_and_policy_survive_restart(self):
        _, result = self.call('/api/improve', {})
        restarted = DemoServer(('127.0.0.1', 0), self.temp.name)
        try:
            self.assertEqual(restarted.policy, Policy(True, 1))
            self.assertEqual(restarted.summary['run_id'], result['run_id'])
            self.assertEqual(len(restarted.summary['cases']), 24)
            self.assertTrue(restarted.summary['applied_to_new_conversations'])
        finally:
            restarted.server_close()

    def test_activation_write_failure_keeps_old_policy_and_exposes_report(self):
        save = Policy.save

        def fail_activation(policy, path):
            if Path(path) == self.server.policy_path:
                raise OSError('Simulated full disk')
            return save(policy, path)

        with patch.object(Policy, 'save', fail_activation), self.assertLogs('scheduler.server', level='ERROR'):
            with self.assertRaises(HTTPError) as error:
                self.call('/api/improve', {})
        self.assertEqual(error.exception.code, 500)
        self.assertEqual(self.server.policy, Policy())
        _, report = self.call('/api/evaluation')
        self.assertTrue(report['gate']['promoted'])
        self.assertFalse(report['applied_to_new_conversations'])
        self.assertTrue(self.server.loop_lock.acquire(blocking=False))
        self.server.loop_lock.release()

    def test_malformed_host_and_origin_are_rejected(self):
        for headers in [{'Host': 'attacker.example'}, {'Host': '127.0.0.1:bad'},
                        {'Origin': self.base + '?unexpected=1'}, {'Origin': self.base + '/path'},
                        {'Origin': 'http://user@' + self.base.removeprefix('http://')}]:
            with self.subTest(headers=headers), self.assertRaises(HTTPError) as error:
                self.call('/api/session', {}, headers)
            self.assertEqual(error.exception.code, 403)

    def test_invalid_request_bodies_and_metadata_are_rejected(self):
        for raw, headers, status in [
            (b'{', {}, 400), (b'\xff', {}, 400), (b'[]', {}, 400),
            (b'x' * 12001, {}, 413), (b'{}', {'Content-Type': 'text/plain'}, 415),
            (b'{}', {'Transfer-Encoding': 'chunked'}, 400),
        ]:
            request = Request(self.base + '/api/session', data=raw,
                              headers={'Content-Type': 'application/json', **headers})
            with self.subTest(status=status, headers=headers):
                with self.assertRaises(HTTPError) as error:
                    urlopen(request, timeout=10)
                self.assertEqual(error.exception.code, status)
        with self.assertRaises(HTTPError) as error:
            self.call('/api/session', {'baseline': 'true'})
        self.assertEqual(error.exception.code, 400)

    def test_invalid_message_ids_and_turns_do_not_mutate_conversation(self):
        _, session = self.call('/api/session', {})
        headers = {'X-Session-ID': session['session_id']}
        for metadata in [{'message_id': '../bad'}, {'message_id': 1},
                         {'expected_turn': True}, {'expected_turn': -1}]:
            with self.subTest(metadata=metadata), self.assertRaises(HTTPError) as error:
                self.call('/api/chat', {'message': 'yes', **metadata}, headers)
            self.assertEqual(error.exception.code, 400)
        self.assertEqual(self.server.sessions[session['session_id']]['agent'].turn, 0)

    def test_report_download_is_fixed_and_static_paths_are_allowlisted(self):
        self.call('/api/improve', {})
        with urlopen(self.base + '/api/report') as response:
            self.assertIn('attachment;', response.headers['Content-Disposition'])
            self.assertIn('87.5%', response.read().decode())
        for path in ['/../runtime/clinic.sqlite3', '/%2e%2e/README.md', '/api/report?file=clinic.sqlite3']:
            if path.startswith('/api/report'):
                with urlopen(self.base + path) as response:
                    self.assertEqual(response.headers.get_content_type(), 'text/markdown')
            else:
                with self.assertRaises(HTTPError) as error:
                    self.call(path)
                self.assertEqual(error.exception.code, 404)

    def wait_for_loop(self):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            _, result = self.call('/api/evaluation')
            if not result.get('running'):
                return result
            time.sleep(0.01)
        self.fail('Background improvement did not finish')

    def test_gemini_job_is_async_and_rejects_duplicate_start(self):
        self.server.gemini_client = GeminiClient(Settings('private-test-key', interval=0))
        entered, release = threading.Event(), threading.Event()
        def blocked(*args, **kwargs):
            entered.set()
            release.wait(5)
            raise GeminiError('Quota reached')
        with patch('scheduler.gemini_learning.improve_gemini', side_effect=blocked), self.assertLogs('scheduler.server', level='ERROR'):
            try:
                status, _ = self.call('/api/improve', {})
                self.assertEqual(status, 202)
                self.assertTrue(entered.wait(2))
                with self.assertRaises(HTTPError) as error:
                    self.call('/api/improve', {})
                self.assertEqual(error.exception.code, 409)
                _, health = self.call('/api/health')
                self.assertEqual(health['mode'], 'gemini')
                self.assertNotIn('private-test-key', json.dumps(health))
            finally:
                release.set()
                result = self.wait_for_loop()
        self.assertIn('Quota', result['loop_error'])
        self.assertEqual(self.server.policy, Policy())
        self.assertFalse(self.server.policy_path.exists())

    def test_gemini_activation_failure_preserves_policy_and_report(self):
        from scheduler.learning import improve
        self.server.gemini_client = GeminiClient(Settings('private-test-key', interval=0))
        save = Policy.save
        def fail_activation(policy, path):
            if Path(path) == self.server.policy_path:
                raise OSError('Disk full')
            return save(policy, path)
        def completed(folder, client, baseline, **kwargs):
            return improve(folder, baseline)
        with patch('scheduler.gemini_learning.improve_gemini', side_effect=completed), patch.object(Policy, 'save', fail_activation), self.assertLogs('scheduler.server', level='ERROR'):
            self.call('/api/improve', {})
            report = self.wait_for_loop()
        self.assertEqual(self.server.policy, Policy())
        self.assertFalse(report['applied_to_new_conversations'])
        self.assertTrue(report['gate']['promoted'])
        self.assertIn('loop_error', report)

    def test_gemini_mode_does_not_display_old_offline_results(self):
        self.call('/api/improve', {})
        restarted = DemoServer(('127.0.0.1', 0), self.temp.name,
                               client=GeminiClient(Settings('private-test-key', interval=0)))
        try:
            self.assertIsNone(restarted.summary)
        finally:
            restarted.server_close()

    def test_initial_comparison_exposes_measured_events_without_old_scores(self):
        self.call('/api/improve', {})
        current = self.server.policy
        self.server.gemini_client = GeminiClient(Settings('private-test-key', interval=0))
        entered, release = threading.Event(), threading.Event()

        def comparison(folder, client, baseline, **kwargs):
            self.assertEqual(baseline, Policy())
            self.assertTrue(kwargs['fresh'])
            kwargs['on_event']({'stage': 'baseline', 'message': 'Training failures found.', 'passed': 21, 'total': 24, 'score': 87.5})
            entered.set()
            release.wait(5)
            from scheduler.learning import improve
            return improve(folder, baseline)

        with patch('scheduler.gemini_learning.improve_gemini', side_effect=comparison):
            try:
                self.assertEqual(self.call('/api/improve', {'baseline': True})[0], 202)
                self.assertTrue(entered.wait(2))
                _, running = self.call('/api/evaluation')
                self.assertTrue(running['running'])
                self.assertTrue(running['pending'])
                self.assertNotIn('after', running)
                self.assertEqual(running['events'][0]['passed'], 21)
                self.assertEqual(self.server.policy, current)
                self.assertNotIn('private-test-key', json.dumps(running))
            finally:
                release.set()
                result = self.wait_for_loop()
        self.assertTrue(result['gate']['promoted'])
        self.assertTrue(result['applied_to_new_conversations'])

    def test_invalid_comparison_setting_does_not_start_an_experiment(self):
        with self.assertRaises(HTTPError) as error:
            self.call('/api/improve', {'baseline': 'initial'})
        self.assertEqual(error.exception.code, 400)
        self.assertFalse(self.server.loop_lock.locked())

    def test_malformed_saved_report_does_not_prevent_startup(self):
        folder = self.server.output_dir
        folder.mkdir(parents=True, exist_ok=True)
        for name in ['summary.json', 'before.json', 'after.json']:
            (folder / name).write_text('[]', encoding='utf-8')
        restarted = DemoServer(('127.0.0.1', 0), self.temp.name)
        try:
            self.assertIsNone(restarted.summary)
            self.assertEqual(restarted.policy, Policy())
        finally:
            restarted.server_close()


if __name__ == "__main__":
    unittest.main()
