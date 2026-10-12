"""Clinic storage and session-scoped scheduling tools."""
from datetime import datetime, timedelta, timezone
import hashlib
import secrets
import sqlite3
import threading

IST = timezone(timedelta(hours=5, minutes=30), "IST")
DOCTORS = {"Dr Rao": "general", "Dr Patel": "general", "Dr Lin": "dermatology"}
PATIENTS = [("p_maya", "Maya Chen", "1990-04-12"), ("p_alex", "Alex Rivera", "1985-09-23")]


class ToolError(Exception):
    code = "TOOL_ERROR"


class TransientError(ToolError):
    code = "READ_TRANSIENT"


class SlotTaken(ToolError):
    code = "SLOT_TAKEN"


class AccessDenied(ToolError):
    code = "ACCESS_DENIED"


class SlotChanged(ToolError):
    code = "SLOT_CHANGED"


class BookingUnavailable(ToolError):
    code = "BOOKING_UNAVAILABLE"


def fingerprint(patient_id, slot):
    fields = (patient_id, slot["id"], slot["doctor"], slot["service"], slot["starts"])
    return hashlib.sha256("|".join(fields).encode()).hexdigest()


class Clinic:
    def __init__(self, path=":memory:", today=None, clock=None):
        self._fixed_day = today
        self._clock = clock
        self._seeded_day = None
        self.lock = threading.RLock()
        self.db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS patients(id TEXT PRIMARY KEY, name TEXT, dob TEXT);
            CREATE TABLE IF NOT EXISTS slots(id TEXT PRIMARY KEY, doctor TEXT, service TEXT, starts TEXT);
            CREATE TABLE IF NOT EXISTS bookings(
                id TEXT PRIMARY KEY, slot_id TEXT UNIQUE REFERENCES slots(id),
                patient_id TEXT REFERENCES patients(id), session_id TEXT, intent_key TEXT UNIQUE,
                confirmation_hash TEXT, confirmed_turn INTEGER, preview_turn INTEGER);
        """)
        with self.lock:
            self.db.executemany("INSERT OR IGNORE INTO patients VALUES(?,?,?)", PATIENTS)
        self.seed_slots()

    @property
    def now(self):
        if self._clock:
            return self._clock().astimezone(IST)
        if self._fixed_day:
            return datetime.combine(self._fixed_day, datetime.min.time(), IST)
        return datetime.now(IST)

    @property
    def today(self):
        return self.now.date()

    def seed_slots(self):
        with self.lock:
            today = self.today
            if self._seeded_day == today:
                return
            for offset in range(15):
                day = today + timedelta(days=offset)
                if day.weekday() >= 5:
                    continue
                for doctor, service in DOCTORS.items():
                    for hour, minute in [(9, 0), (10, 30), (14, 0), (16, 0)]:
                        start = datetime.combine(day, datetime.min.time(), IST).replace(hour=hour, minute=minute)
                        slot_id = f"{day.isoformat()}_{doctor.split()[-1].lower()}_{hour:02}{minute:02}"
                        self.db.execute("INSERT OR IGNORE INTO slots VALUES(?,?,?,?)", (slot_id, doctor, service, start.isoformat()))
            self._seeded_day = today

    def close(self):
        with self.lock:
            self.db.close()

    def bookings(self, session_id=None):
        with self.lock:
            query = "SELECT b.*, s.doctor, s.service, s.starts FROM bookings b JOIN slots s ON b.slot_id=s.id"
            args = ()
            if session_id:
                query += " WHERE session_id=?"
                args = (session_id,)
            return [dict(row) for row in self.db.execute(query, args)]

    def occupy(self, slot_id):
        """Evaluation fault: another patient books between preview and commit."""
        with self.lock:
            self.db.execute("INSERT INTO bookings VALUES(?,?,?,?,?,?,?,?)", (
                "external_" + secrets.token_hex(4), slot_id, "p_alex", "external", secrets.token_hex(8), "external", 1, 0))


class ToolGateway:
    def __init__(self, clinic, session_id):
        self.clinic = clinic
        self.session_id = session_id
        self.patient_id = None
        self.events = []
        self.faults = {}
        self.turn = 0
        self._preview = None
        self._approved = None

    def _event(self, tool, outcome, **safe_fields):
        self.events.append({"tool": tool, "outcome": outcome, "turn": self.turn, **safe_fields})

    def _fail_if_requested(self, tool):
        remaining = self.faults.get(tool, 0)
        if remaining:
            self.faults[tool] = remaining - 1
            self._event(tool, "READ_TRANSIENT")
            raise TransientError("Clinic service temporarily unavailable")

    def verify_patient(self, name, dob):
        self._fail_if_requested("verify_patient")
        try:
            with self.clinic.lock:
                row = self.clinic.db.execute("SELECT id FROM patients WHERE lower(name)=lower(?) AND dob=?", (name.strip(), dob)).fetchone()
        except sqlite3.OperationalError as error:
            self._event("verify_patient", "READ_TRANSIENT")
            raise TransientError("Patient verification is unavailable") from error
        self.patient_id = row["id"] if row else None
        self.invalidate_preview()
        self._event("verify_patient", "verified" if row else "not_verified")
        return bool(row)

    def search_slots(self, request):
        if not self.patient_id:
            self._event("search_slots", "ACCESS_DENIED")
            raise AccessDenied("Verify the patient first")
        self._fail_if_requested("search_slots")
        try:
            self.clinic.seed_slots()
            with self.clinic.lock:
                rows = self.clinic.db.execute("""SELECT s.* FROM slots s
                    LEFT JOIN bookings b ON s.id=b.slot_id
                    WHERE b.id IS NULL AND substr(s.starts,1,10)=? AND s.service=? AND s.starts>?
                    ORDER BY s.starts,s.doctor""", (request.day, request.service, self.clinic.now.isoformat())).fetchall()
        except sqlite3.OperationalError as error:
            self._event("search_slots", "READ_TRANSIENT")
            raise TransientError("Availability is unavailable") from error
        slots = [dict(r) for r in rows if not request.doctor or r["doctor"] == request.doctor]
        if request.hard_period:
            slots = [s for s in slots if (datetime.fromisoformat(s["starts"]).hour >= 12) == (request.hard_period == "afternoon")]
        if request.exact_time:
            slots = [s for s in slots if s["starts"][11:16] == request.exact_time]
        self._event("search_slots", "ok", count=len(slots))
        return slots

    def preview(self, slot, intent_key):
        if not self.patient_id:
            raise AccessDenied("Verify the patient first")
        # Re-read canonical fields: no patient-controlled slot data is trusted.
        with self.clinic.lock:
            row = self.clinic.db.execute("SELECT * FROM slots WHERE id=?", (slot["id"],)).fetchone()
        if not row or dict(row) != slot:
            raise AccessDenied("Invalid slot preview")
        if datetime.fromisoformat(row["starts"]) <= self.clinic.now:
            raise SlotChanged("This appointment has already started")
        self._preview = {"slot": dict(row), "hash": fingerprint(self.patient_id, row), "turn": self.turn, "key": intent_key}
        self._approved = None
        self._event("preview_booking", "ok", slot_id=slot["id"], fingerprint=self._preview["hash"])

    def invalidate_preview(self):
        self._preview = self._approved = None

    def approve(self):
        if not self._preview or self.turn <= self._preview["turn"]:
            raise AccessDenied("A later patient confirmation is required")
        self._approved = {**self._preview, "confirmed_turn": self.turn}

    def find_booking(self):
        if not self.patient_id or not self._approved:
            raise AccessDenied("No confirmed request to check")
        try:
            with self.clinic.lock:
                row = self.clinic.db.execute("""SELECT b.id, s.id AS slot_id, s.doctor, s.service, s.starts
                    FROM bookings b JOIN slots s ON b.slot_id=s.id
                    WHERE b.intent_key=? AND b.session_id=? AND b.patient_id=?""",
                    (self._approved["key"], self.session_id, self.patient_id)).fetchone()
        except sqlite3.OperationalError as error:
            self._event("check_booking", "READ_TRANSIENT")
            raise TransientError("Booking status is unavailable") from error
        self._event("check_booking", "booked" if row else "not_booked")
        return dict(row) if row else None

    def book(self):
        grant = self._approved
        if not self.patient_id or not grant or grant["hash"] != fingerprint(self.patient_id, grant["slot"]):
            self._event("book_appointment", "ACCESS_DENIED")
            raise AccessDenied("Verified identity and exact preview confirmation required")
        with self.clinic.lock:
            try:
                self.clinic.db.execute("BEGIN IMMEDIATE")
                existing = self.clinic.db.execute("SELECT id,slot_id,patient_id,session_id FROM bookings WHERE intent_key=?", (grant["key"],)).fetchone()
                if existing:
                    if existing["slot_id"] != grant["slot"]["id"] or existing["patient_id"] != self.patient_id or existing["session_id"] != self.session_id:
                        raise AccessDenied("Idempotency key cannot change slots")
                    booking_id = existing["id"]
                else:
                    canonical = self.clinic.db.execute("SELECT * FROM slots WHERE id=?", (grant["slot"]["id"],)).fetchone()
                    if not canonical or fingerprint(self.patient_id, canonical) != grant["hash"] or datetime.fromisoformat(canonical["starts"]) <= self.clinic.now:
                        raise SlotChanged("Slot details changed or the appointment already started")
                    booking_id = "CL-" + secrets.token_hex(4).upper()
                    self.clinic.db.execute("INSERT INTO bookings VALUES(?,?,?,?,?,?,?,?)", (
                        booking_id, grant["slot"]["id"], self.patient_id, self.session_id, grant["key"],
                        grant["hash"], grant["confirmed_turn"], grant["turn"]))
                self.clinic.db.execute("COMMIT")
            except sqlite3.IntegrityError as exc:
                self.clinic.db.execute("ROLLBACK")
                occupied = self.clinic.db.execute("SELECT 1 FROM bookings WHERE slot_id=?", (grant["slot"]["id"],)).fetchone()
                if not occupied:
                    self._event("book_appointment", "BOOKING_UNAVAILABLE")
                    raise BookingUnavailable("Booking could not be committed") from exc
                self._event("book_appointment", "SLOT_TAKEN", slot_id=grant["slot"]["id"])
                self.invalidate_preview()
                raise SlotTaken("That slot was just taken") from exc
            except sqlite3.OperationalError as error:
                if self.clinic.db.in_transaction:
                    self.clinic.db.execute("ROLLBACK")
                self._event("book_appointment", "BOOKING_UNAVAILABLE")
                raise BookingUnavailable("Booking result must be checked") from error
            except SlotChanged:
                self.clinic.db.execute("ROLLBACK")
                self._event("book_appointment", "SLOT_CHANGED", slot_id=grant["slot"]["id"])
                self.invalidate_preview()
                raise
            except Exception:
                if self.clinic.db.in_transaction:
                    self.clinic.db.execute("ROLLBACK")
                raise
        self._event("book_appointment", "booked", booking_id=booking_id, slot_id=grant["slot"]["id"],
                    patient_verified=True, fingerprint=grant["hash"], preview_turn=grant["turn"], confirmed_turn=grant["confirmed_turn"])
        return {**grant["slot"], "slot_id": grant["slot"]["id"], "id": booking_id}
