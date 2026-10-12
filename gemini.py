"""Small server-side Gemini client. Credentials never enter browser responses."""
from dataclasses import dataclass, field, replace
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time
from contextlib import contextmanager
from urllib import error, request

ROOT = Path(__file__).resolve().parents[1]


class GeminiError(ValueError):
    pass


@dataclass(frozen=True)
class Settings:
    api_key: str = field(repr=False)
    model: str = 'gemini-3.5-flash-lite'
    max_calls: int = 180
    interval: float = 15
    repair_model: str = ''

    @classmethod
    def load(cls, path=None):
        values = {}
        file = Path(path) if path else ROOT / '.env'
        if file.exists():
            for line in file.read_text(encoding='utf-8-sig').splitlines():
                if '=' in line and not line.lstrip().startswith('#'):
                    key, value = line.split('=', 1)
                    values[key.strip()] = value.strip().strip('\"\'')
        def get(name, default=''):
            return os.environ.get(name, values.get(name, default))
        key = get('GEMINI_API_KEY')
        if not key or key in {'your_key_here', 'paste_your_key_here'}:
            raise GeminiError('Add GEMINI_API_KEY to the local .env file before using Gemini.')
        model = get('GEMINI_MODEL', 'gemini-3.5-flash-lite')
        if not re.fullmatch(r'[a-z0-9][a-z0-9.-]{1,99}', model):
            raise GeminiError('GEMINI_MODEL must be a model name, not a URL.')
        repair_model = get('GEMINI_REPAIR_MODEL') or model
        if not re.fullmatch(r'[a-z0-9][a-z0-9.-]{1,99}', repair_model):
            raise GeminiError('GEMINI_REPAIR_MODEL must be a model name, not a URL.')
        try:
            calls = int(get('GEMINI_MAX_CALLS', '180'))
            interval = float(get('GEMINI_REQUEST_INTERVAL', '15'))
            if not 1 <= calls <= 500 or not 0 <= interval <= 30:
                raise ValueError()
        except ValueError:
            raise GeminiError('Use 1–500 for GEMINI_MAX_CALLS and 0–30 seconds for GEMINI_REQUEST_INTERVAL.') from None
        return cls(key, model, calls, interval, repair_model)


class NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class GeminiClient:
    def __init__(self, settings=None, cache_path=None, fresh=False):
        self.settings = settings or Settings.load()
        self.calls = self.cache_hits = self.tokens = 0
        self._cache = {}
        self.cache_path = Path(cache_path) if cache_path else None
        if self.cache_path and self.cache_path.exists() and not fresh:
            try:
                if self.cache_path.stat().st_size > 2_000_000:
                    raise ValueError('Oversized cache')
                saved = json.loads(self.cache_path.read_text(encoding='utf-8'))
                entries = saved['responses']
                if saved['model'] != self.model or not isinstance(entries, dict) or len(entries) > 500:
                    raise ValueError('Incompatible cache')
                for key, value in entries.items():
                    if not re.fullmatch(r'[0-9a-f]{64}', key) or not isinstance(value, str) or not isinstance(json.loads(value), dict):
                        raise ValueError('Invalid cache')
                self._cache = entries
            except (OSError, ValueError, KeyError, TypeError):
                self._cache = {}
        self._lock = threading.Lock()
        self._last = 0.0
        self._opener = request.build_opener(NoRedirect())

    @property
    def model(self):
        return self.settings.model

    def stats(self):
        return {'model': self.model, 'requests': self.calls, 'cache_hits': self.cache_hits,
                'reported_tokens': self.tokens, 'request_limit': self.settings.max_calls}

    def repair_client(self):
        model = self.settings.repair_model or self.model
        return self if model == self.model else GeminiClient(replace(self.settings, model=model))

    def _request(self, encoded):
        for attempt in range(3):
            if self.calls >= self.settings.max_calls:
                raise GeminiError('Gemini request budget reached. No additional requests were sent.')
            time.sleep(max(0, self.settings.interval - (time.monotonic() - self._last)))
            self.calls += 1
            self._last = time.monotonic()
            req = request.Request(
                f'https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent',
                data=encoded, headers={'Content-Type': 'application/json', 'x-goog-api-key': self.settings.api_key})
            try:
                with self._opener.open(req, timeout=25) as response:
                    return response.read(2_000_001)
            except error.HTTPError as exc:
                if exc.code not in {500, 502, 503, 504} or attempt == 2 or self.calls >= self.settings.max_calls:
                    raise
            except (error.URLError, TimeoutError, OSError):
                if attempt == 2 or self.calls >= self.settings.max_calls:
                    raise
            time.sleep(2 ** attempt)

    @contextmanager
    def _available(self):
        if not self._lock.acquire(blocking=False):
            raise GeminiError('Gemini is handling another request. Please retry in a moment.')
        try:
            yield
        finally:
            self._lock.release()

    def generate(self, instruction, data, schema, *, cache=True):
        body = {'systemInstruction': {'parts': [{'text': instruction}]},
                'contents': [{'role': 'user', 'parts': [{'text': json.dumps(data, ensure_ascii=False)}]}],
                'generationConfig': {'temperature': 0, 'maxOutputTokens': 8192,
                                     'responseMimeType': 'application/json', 'responseSchema': schema}}
        encoded = json.dumps(body).encode()
        digest = hashlib.sha256(encoded).hexdigest()
        with self._available():
            if cache and digest in self._cache:
                self.cache_hits += 1
                return json.loads(self._cache[digest])
            try:
                raw = self._request(encoded)
                if len(raw) > 2_000_000:
                    raise GeminiError('Gemini response exceeded the size limit.')
                result = json.loads(raw)
                candidate = result['candidates'][0]
                if candidate.get('finishReason') != 'STOP':
                    raise GeminiError('Gemini did not finish a valid response. No change was applied.')
                content = ''.join(p.get('text', '') for p in candidate['content']['parts'] if not p.get('thought'))
                value = json.loads(content)
                if not isinstance(value, dict):
                    raise ValueError()
            except error.HTTPError as exc:
                messages = {429: 'Gemini free-tier quota or rate limit reached. Wait for quota to reset; no paid fallback is used.',
                            503: 'Gemini is temporarily overloaded (HTTP 503). Retry later; evaluation progress is saved.',
                            400: 'Gemini rejected the request or model configuration.',
                            403: 'Gemini key or project does not permit this request.',
                            404: 'The configured Gemini model is unavailable for this key.'}
                raise GeminiError(messages.get(exc.code, f'Gemini returned HTTP {exc.code}. No change was applied.')) from None
            except (error.URLError, TimeoutError, OSError):
                raise GeminiError('Gemini could not be reached. Check the connection and try again.') from None
            except GeminiError:
                raise
            except (ValueError, KeyError, IndexError, TypeError, AttributeError):
                raise GeminiError('Gemini returned invalid structured data. No change was applied.') from None
            usage = result.get('usageMetadata')
            tokens = usage.get('totalTokenCount', 0) if isinstance(usage, dict) else 0
            if type(tokens) is int and tokens >= 0:
                self.tokens += tokens
            if cache:
                self._cache[digest] = json.dumps(value)
                if len(self._cache) > 500:
                    del self._cache[next(iter(self._cache))]
                if self.cache_path:
                    from .storage import write_json
                    write_json(self.cache_path, {'model': self.model, 'responses': self._cache})
            return value
