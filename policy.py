"""The only behavior the learning loop is allowed to change."""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from .storage import write_json


@dataclass(frozen=True)
class Policy:
    honor_soft_preferences: bool = False
    read_retries: int = 0
    interpretation_notes: str = ""

    def __post_init__(self):
        if type(self.honor_soft_preferences) is not bool:
            raise ValueError("honor_soft_preferences must be boolean")
        if type(self.read_retries) is not int or not 0 <= self.read_retries <= 2:
            raise ValueError("read_retries must be an integer between 0 and 2")
        if not isinstance(self.interpretation_notes, str) or len(self.interpretation_notes) > 4000:
            raise ValueError("interpretation_notes must be text of at most 4000 characters")

    @property
    def version(self):
        values = asdict(self)
        if not self.interpretation_notes:
            values.pop('interpretation_notes')
        body = json.dumps(values, sort_keys=True).encode()
        return hashlib.sha256(body).hexdigest()[:12]

    def to_dict(self):
        return {**asdict(self), "version": self.version}

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ValueError("Policy must be a JSON object")
        extra = set(value) - {"honor_soft_preferences", "read_retries", "interpretation_notes", "version"}
        if extra:
            raise ValueError(f"Unknown policy fields: {sorted(extra)}")
        required = {"honor_soft_preferences", "read_retries"}
        if not required <= set(value):
            raise ValueError("Policy must include both behavior fields")
        result = cls(**{k: value[k] for k in required}, interpretation_notes=value.get('interpretation_notes', ''))
        if "version" in value and value["version"] != result.version:
            raise ValueError("Policy version does not match content")
        return result

    @classmethod
    def load(cls, path):
        file = Path(path)
        return cls.from_dict(json.loads(file.read_text(encoding="utf-8"))) if file.exists() else cls()

    def save(self, path):
        write_json(path, self.to_dict())
