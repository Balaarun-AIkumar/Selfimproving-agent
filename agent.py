"""Conversation state and the scheduling workflow."""
from dataclasses import asdict
from datetime import datetime
import re
import secrets

from .clinic import AccessDenied, ToolGateway, TransientError, SlotTaken, SlotChanged, BookingUnavailable
from .parsing import Clarification, Request, parse_request
from .policy import Policy


def redact(text):
    text = re.sub(r"(\b(?:DOB|date of birth|born)\s*[:=,]?\s*)\d{4}-\d{2}-\d{2}", r"\1[DOB redacted]", text, flags=re.I)
    return re.sub(r"\b(?:19|20)\d{2}-\d{2}-\d{2}\b", lambda m: "[DOB redacted]" if int(m[0][:4]) < 2020 else m[0], text)


class Agent:
    greeting = "Hello, I can help schedule general care or dermatology. Which doctor and date would you like? Use the fictional patient details in this demo."

    def __init__(self, clinic, policy=None, session_id=None, interpreter=None):
        self.policy = policy or Policy()
        self.interpreter = interpreter
        self.model_events = []
        self.model_errors = []
        self.session_id = session_id or secrets.token_hex(16)
        self.tools = ToolGateway(clinic, self.session_id)
        self.request = Request()
        self.name = self.dob = None
        self.identity_attempts = 0
        self.turn = 0
        self.intent = 0
        self.stage = "identity"
        self.offers = []
        self.selected = None
        self.receipt = None
        self.pending_fields = set()
        self.history = [{"role": "assistant", "text": self.greeting}]

    def _read(self, operation, *args):
        for attempt in range(self.policy.read_retries + 1):
            try:
                return operation(*args)
            except TransientError:
                if attempt == self.policy.read_retries:
                    raise

    @staticmethod
    def _urgent(text):
        text = text.replace("’", "'")
        text = re.sub(r"\b(?:no|without|denies|do not have|don't have)\s+(?:any\s+)?chest pain\b", "", text)
        text = re.sub(r"\b(?:no|without|denies)\s+(?:any\s+)?(?:difficulty breathing|shortness of breath)\b", "", text)
        return bool(re.search(r"\b(?:chest pain|severe difficulty breathing|can't breathe|cannot breathe|medical emergency|i am having an emergency)\b", text))

    def _update_request(self, text):
        try:
            candidate, supplied = parse_request(text, self.request, self.tools.clinic.today)
        except Clarification as error:
            self.pending_fields.update(error.fields)
            if "timezone" in error.fields:
                self.pending_fields.update({"day", "time"})
            for field in self.pending_fields:
                if field == "time":
                    self.request.hard_period = self.request.soft_period = self.request.exact_time = None
                elif field in {"day", "doctor", "service"}:
                    setattr(self.request, field, None)
            self._clear_selection()
            return False, str(error)
        changed = candidate != self.request
        self.request = candidate
        if "timezone" in self.pending_fields and not {"timezone", "day", "time"} <= supplied:
            supplied.discard("timezone")
        self.pending_fields.difference_update(supplied)
        if changed or supplied:
            self.intent += 1
            self._clear_selection()
        return changed or bool(supplied), None

    def _clear_selection(self):
        self.selected = None
        self.offers = []
        self.tools.invalidate_preview()

    def _identity(self, original):
        text = original
        dob_match = re.search(r"\b(?:dob|date of birth|born)\s*[:=,]?\s*(\d{4}-\d{2}-\d{2})", text, re.I)
        if not dob_match and self.name and re.fullmatch(r"\s*\d{4}-\d{2}-\d{2}[.!]?\s*", text):
            dob_match = re.search(r"(\d{4}-\d{2}-\d{2})", text)
        if dob_match:
            self.dob = dob_match[1]
            text = text[:dob_match.start()] + text[dob_match.end():]
        name_match = re.search(r"\b(?:my name is|name[:=])\s*([A-Za-z]+(?:[ '\-][A-Za-z]+){1,3}?)(?=\s*[,.;]|\s+(?:dob|date of birth)|$)", original, re.I)
        if name_match:
            self.name = name_match[1].strip()
        else:
            for name in ["Maya Chen", "Alex Rivera"]:
                if name.lower() in text.lower():
                    self.name = name
        return text

    def _offer(self):
        if self.pending_fields:
            self.stage = "clarify"
            if "timezone" in self.pending_fields:
                return "Please restate the appointment date and time explicitly in IST before choosing an appointment. Nothing has been booked."
            labels = {"day": "appointment date", "time": "appointment time", "doctor": "doctor", "service": "service", "timezone": "date and time explicitly in IST"}
            needed = ", ".join(labels[f] for f in sorted(self.pending_fields))
            return f"Please clarify the {needed} before choosing an appointment. Nothing has been booked."
        if not self.request.service:
            self.stage = "request"
            return "Would you like general care (Dr Rao or Dr Patel) or dermatology (Dr Lin)?"
        if not self.request.day:
            self.stage = "request"
            return "What appointment date would you like? Use YYYY-MM-DD or ‘tomorrow’. All clinic times are IST (UTC+05:30)."
        try:
            slots = self._read(self.tools.search_slots, self.request)
        except TransientError:
            self.stage = "service_unavailable"
            return "The clinic availability service is unavailable. Nothing has been booked. Please try again or contact the clinic team."
        if self.policy.honor_soft_preferences and self.request.soft_period:
            target_afternoon = self.request.soft_period == "afternoon"
            slots.sort(key=lambda s: ((datetime.fromisoformat(s["starts"]).hour >= 12) != target_afternoon, s["starts"], s["doctor"]))
        self.offers = slots[:3]
        self.stage = "choose" if self.offers else "request"
        if not self.offers:
            return "No available slots match those details. Nothing has been booked. Would you like a different date, doctor, or time?"
        lines = [f"{i}. {s['doctor']} · {s['service']} · {s['starts'][:10]} {s['starts'][11:16]} IST (UTC+05:30)" for i, s in enumerate(self.offers, 1)]
        return "Available appointments:\n" + "\n".join(lines) + "\nChoose an option number. I’ll show a confirmation before booking."

    def _receipt_text(self):
        s = self.receipt
        return f"Booked. Reference {s['id']}: {s['doctor']}, {s['service']}, {s['starts'][:10]} at {s['starts'][11:16]} IST (UTC+05:30). The clinic record confirms this appointment."

    def _respond(self, original):
        text = original.lower().strip()
        if self._urgent(text):
            self.tools.invalidate_preview()
            self.stage = "escalated"
            return "Your message may describe an emergency. Contact your local emergency service now. I can’t assess symptoms or continue routine scheduling. Any existing booking stays in the clinic record; contact the clinic team to check it."
        if re.search(r"\b(?:ignore|override|bypass)\b.*\b(?:instructions|rules|verification|confirmation)\b|\b(?:reveal|list|show)\b.*\b(?:other patients|all patients|patient records)\b|\bbook without\b", text):
            return "I can only schedule for a verified patient, using available clinic slots and explicit confirmation. I can’t reveal other patient records or bypass those requirements."
        if self.stage in {"escalated", "cancelled", "handoff"}:
            return "This conversation is closed. Please contact the clinic team for human help, or start a new demo conversation."
        if self.stage == "booked":
            if "cancel" in text or "change" in text:
                return "Your appointment remains booked. This demo does not cancel or reschedule existing bookings; contact the clinic team and quote " + self.receipt["id"] + "."
            return self._receipt_text()
        if self.stage == "booking_uncertain":
            try:
                receipt = self.tools.find_booking()
            except TransientError:
                return "The clinic still can’t confirm the result. Please try again or contact the clinic team. Don’t start another booking until this request is checked."
            if receipt:
                self.receipt = receipt
                self.stage = "booked"
                return self._receipt_text()
            self.stage = "confirm"
        if self.tools.patient_id and re.search(r"\b(?:name\s*:|my name is|dob|date of birth|born)\b", text):
            self._clear_selection()
            self.stage = "handoff"
            return "This conversation is already verified for one patient. Start a new conversation to change patient details. Nothing new has been booked."
        if re.fullmatch(r"(?:cancel|stop|never mind|nevermind|cancel request)[.!]?", text):
            self.tools.invalidate_preview()
            self.stage = "cancelled"
            return "Scheduling request cancelled. Nothing has been booked."
        if re.search(r"\b(?:diagnose|diagnosis|prescribe|dosage|medical advice)\b", text):
            return "I can help with scheduling, but I can’t diagnose, prescribe, or provide medical advice. Please ask a clinician. Would you like to schedule a visit?"
        request_text = original
        if not self.tools.patient_id:
            request_text = self._identity(original)
        if self.interpreter:
            from .language import local_control
            from .gemini import GeminiError
            if not local_control(original):
                try:
                    decision = self.interpreter.interpret(request_text, self)
                    self.model_events.append({'turn': self.turn, **decision})
                except GeminiError as error:
                    self.model_errors.append({'turn': self.turn, 'error': str(error)})
                    if self.stage in {'choose', 'confirm'}:
                        self._clear_selection()
                        self.pending_fields.update({'day', 'time'})
                        self.stage = 'clarify'
                    return str(error) + ' Please retry your request. No new booking was made.'
                action = decision['action']
                if action == 'urgent':
                    return self._respond('medical emergency')
                if action == 'cancel':
                    self._clear_selection()
                    self.stage = 'cancelled'
                    return 'Scheduling request cancelled. Nothing has been booked.'
                if action == 'medical':
                    return 'I can help schedule a visit, but medical advice requires a clinician.'
                if action == 'decline':
                    self._clear_selection()
                    return self._offer() if self.tools.patient_id else 'Please provide your demo patient details first.'
                if action == 'clarify':
                    _, original_error = self._update_request(request_text.lower())
                    if original_error:
                        self.stage = 'clarify' if self.tools.patient_id else 'identity'
                        return original_error
                    self._clear_selection()
                    self.pending_fields.update({'day', 'time'})
                    self.stage = 'clarify' if self.tools.patient_id else 'identity'
                    return 'Please clarify the appointment date and time in IST, and name a supported doctor or service. Nothing has been booked.'
                if action == 'update':
                    request_text = decision['canonical_request']
                elif action == 'choose':
                    text = str(decision['choice'])
                    request_text = ''
        changed, error = self._update_request(request_text.lower())
        if error:
            self.stage = "clarify" if self.tools.patient_id else "identity"
            return error
        if not self.tools.patient_id:
            self.stage = "identity"
            if not self.name or not self.dob:
                return "Please provide your full name and date of birth as ‘Name: …, DOB: YYYY-MM-DD’. Use one of the fictional patient records in this demo." if not self.name else "Please provide your date of birth as DOB: YYYY-MM-DD so I can verify your demo patient record."
            try:
                verified = self._read(self.tools.verify_patient, self.name, self.dob)
            except TransientError:
                self.stage = "service_unavailable"
                return "The patient verification service is unavailable. Nothing has been booked. Please try again or contact the clinic team."
            if not verified:
                self.identity_attempts += 1
                self.dob = None
                if self.identity_attempts >= 3:
                    self.stage = "handoff"
                    return "I couldn’t verify your identity after three attempts. Nothing has been booked. Please contact the clinic team for human verification."
                return "I couldn’t verify those details. Please check your full name and DOB, or contact the clinic team. Nothing has been booked."
            return "Demo patient verified. " + self._offer()
        if changed or self.stage in {"request", "identity", "service_unavailable", "clarify"}:
            return self._offer()
        if self.stage == "choose":
            choice = re.fullmatch(r"(?:option\s*)?([1-3])[.!]?|(?:the\s+)?(first|second|third)(?:\s+(?:one|option))?[.!]?", text)
            if not choice:
                return "Please choose one of the displayed option numbers, or tell me what date, doctor, or time you want to change."
            index = int(choice[1]) - 1 if choice[1] else ["first", "second", "third"].index(choice[2])
            if index >= len(self.offers):
                return "That option isn’t available. Please choose one of the displayed option numbers."
            self.selected = self.offers[index]
            try:
                self.tools.preview(self.selected, f"{self.session_id}:{self.intent}")
            except (SlotChanged, AccessDenied):
                self._clear_selection()
                return "That appointment changed or has already started. Nothing has been booked. " + self._offer()
            self.stage = "confirm"
            s = self.selected
            return f"Please confirm: {s['doctor']}, {s['service']}, {s['starts'][:10]} at {s['starts'][11:16]} IST (UTC+05:30), for your verified demo patient record. Reply ‘yes’ to book this exact appointment, ‘no’ to choose again, or give changed details. Nothing is booked yet."
        if self.stage == "confirm":
            if re.fullmatch(r"(?:no|no thanks|decline)[.!]?", text):
                self.tools.invalidate_preview()
                self.selected = None
                return self._offer()
            if not re.fullmatch(r"(?:yes|yes please|confirm|please confirm|book it|yes book it)[.!]?", text):
                return "Please reply ‘yes’ to confirm this exact appointment, ‘no’ to choose again, or give the details you want to change. Nothing is booked yet."
            self.tools.approve()
            try:
                self.receipt = self.tools.book()
            except SlotTaken:
                self.selected = None
                return "That slot was just taken by another patient. Nothing has been booked for you. " + self._offer()
            except SlotChanged:
                self._clear_selection()
                return "That appointment changed or has already started. Nothing has been booked for you. " + self._offer()
            except BookingUnavailable:
                self.stage = "booking_uncertain"
                return "The clinic couldn’t confirm the booking result. Reply ‘retry’ to check this same request before doing anything else."
            self.stage = "booked"
            return self._receipt_text()
        return self._offer()

    def reply(self, message):
        if not isinstance(message, str) or not message.strip() or len(message) > 2000:
            raise ValueError("Enter a message between 1 and 2000 characters")
        if self.turn >= 120:
            raise ValueError("This conversation has reached its limit. Start a new conversation; existing bookings remain unchanged.")
        self.turn += 1
        self.tools.turn = self.turn
        self.history.append({"role": "user", "text": message})
        try:
            response = self._respond(message)
        except Exception:
            self.stage = "booking_uncertain" if self.tools._approved else "handoff"
            self.history.append({"role": "assistant", "text": "The clinic service failed. Check this request before starting another booking."})
            raise
        self.history.append({"role": "assistant", "text": response})
        return response

    def snapshot(self):
        return {"session_id": self.session_id, "turn": self.turn, "stage": self.stage, "policy": self.policy.to_dict(),
                "language_model": self.interpreter.model if self.interpreter else None,
                "model_events": self.model_events, "model_errors": self.model_errors,
                "request": asdict(self.request), "verified": bool(self.tools.patient_id),
                "patient_name": self.name if self.tools.patient_id else None,
                "offers": self.offers if self.stage == "choose" else [],
                "selected": self.selected if self.stage == "confirm" else None,
                "needs_clarification": sorted(self.pending_fields),
                "history": [{**turn, "text": redact(turn["text"])} for turn in self.history],
                "tools": self.tools.events, "receipt": self.receipt}
