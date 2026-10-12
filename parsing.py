"""Parse the scheduling fields we support; ask about everything ambiguous."""
from dataclasses import asdict, dataclass
from datetime import date, timedelta
import re

from .clinic import DOCTORS


@dataclass
class Request:
    day: str | None = None
    service: str | None = None
    doctor: str | None = None
    hard_period: str | None = None
    soft_period: str | None = None
    exact_time: str | None = None


class Clarification(ValueError):
    def __init__(self, message, fields):
        super().__init__(message)
        self.fields = set(fields)


def parse_request(text, current, today):
    text = text.lower().replace("’", "'")
    values = asdict(current)
    supplied = set()

    # Remove the supported timezone before looking for clock times.
    supported_timezone = bool(re.search(r"\butc\+0?5:30\b|\basia/kolkata\b", text))
    text = re.sub(r"\butc\+0?5:30\b|\basia/kolkata\b", "", text)
    if re.search(r"\b(?:pst|pdt|est|edt|cst|cdt|mst|mdt|cet|cest|gmt|bst|utc|jst|kst|aest|aedt|acst|nzst|nzdt|hst|london|new york)\b|\b[a-z]+/[a-z_]+\b|\b(?:eastern|pacific|central|mountain)\s+time\b", text):
        raise Clarification("All appointments use IST (UTC+05:30). I won’t guess a timezone conversion. Please restate the appointment date and time in IST.", {"timezone"})
    if supported_timezone or re.search(r"\b(?:ist|kolkata|india)\b", text):
        supplied.add("timezone")

    doctors = {d for d in DOCTORS if re.search(r"\b(?:dr\.?\s+)?" + d.split()[-1].lower() + r"\b", text)}
    doctor_text = re.sub(r"\b(?:any doctor|no doctor preference)\b", "", text)
    doctor_tokens = re.findall(r"\bdr\.?\s+([a-z]+)\b", doctor_text)
    if any(token not in {d.split()[-1].lower() for d in DOCTORS} for token in doctor_tokens):
        raise Clarification("That doctor is not in this demo clinic. Choose Dr Rao or Dr Patel for general care, or Dr Lin for dermatology.", {"doctor"})
    if len(doctors) > 1:
        raise Clarification("Please choose one doctor, or say ‘any doctor’.", {"doctor"})
    any_doctor = bool(re.search(r"\b(?:any doctor|no doctor preference)\b", text))
    if doctors and any_doctor:
        raise Clarification("Would you like the named doctor, or any available doctor?", {"doctor"})
    services = set()
    if re.search(r"\b(?:cardiology|neurology|pediatrics|paediatrics|dentist|dental|orthopedics|orthopaedics|psychiatry)\b", text):
        raise Clarification("This demo clinic offers general care and dermatology. Please choose one of those services.", {"service"})
    if re.search(r"\b(?:dermatology|skin)\b", text):
        services.add("dermatology")
    if re.search(r"\b(?:general|checkup|check-up|gp)\b", text) or (not services and re.search(r"\broutine\b", text)):
        services.add("general")
    if len(services) > 1:
        raise Clarification("Please choose general care or dermatology for this appointment.", {"service"})
    doctor = next(iter(doctors), None)
    service = next(iter(services), None)
    if doctor and service and DOCTORS[doctor] != service:
        raise Clarification(f"{doctor} provides {DOCTORS[doctor]} care. Would you like that doctor’s service, or a doctor who provides {service} care?", {"doctor", "service"})
    if any_doctor:
        values["doctor"] = None
        supplied.add("doctor")
    if service:
        values["service"] = service
        supplied.add("service")
        if values["doctor"] and DOCTORS[values["doctor"]] != service:
            values["doctor"] = None
    if doctor:
        values.update(doctor=doctor, service=DOCTORS[doctor])
        supplied.update({"doctor", "service"})

    iso_dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", text)
    without_iso = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", "", text)
    if re.search(r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b", without_iso):
        raise Clarification("That date is ambiguous. Please give the appointment date as YYYY-MM-DD, in IST.", {"day"})
    days = []
    for value in iso_dates:
        try:
            days.append(date.fromisoformat(value))
        except ValueError:
            raise Clarification("That calendar date is invalid. Please use a valid YYYY-MM-DD appointment date.", {"day"}) from None
    if re.search(r"\btomorrow\b", text):
        days.append(today + timedelta(days=1))
    if re.search(r"\btoday\b", text):
        days.append(today)
    for weekday, name in enumerate(["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]):
        if re.search(r"\b" + name + r"\b", text):
            days.append(today + timedelta(days=(weekday - today.weekday()) % 7 or 7))
    if len(set(days)) > 1:
        raise Clarification("I found more than one appointment date. Please choose one date as YYYY-MM-DD.", {"day"})
    if days:
        if days[0] < today:
            raise Clarification("I can’t schedule an appointment in the past. Please choose a future date.", {"day"})
        values["day"] = days[0].isoformat()
        supplied.add("day")
    elif re.search(r"\b(?:next week|next month|tonight|yesterday)\b|\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b", text):
        raise Clarification("Please give that appointment date as YYYY-MM-DD or a named weekday.", {"day"})

    periods = set(re.findall(r"\b(morning|afternoon)s?\b", text))
    if re.search(r"\b(?:after|before|between|not(?: at)?|no earlier than|no later than)\b.*\b\d{1,2}(?::\d{2}|\s*[ap]m)\b|\b\d{1,2}\s+(?:to|until)\s+\d{1,2}(?:\s*[ap]m)?\b|\b(?:not|no|can't do|cannot do)\s+(?:the\s+)?(?:mornings?|afternoons?)\b", text):
        raise Clarification("Please clarify the time as morning only, afternoon only, or an exact HH:MM time in IST. Time ranges and negated preferences need clarification.", {"time"})
    if len(periods) > 1:
        raise Clarification("Would you like morning, afternoon, or any time?", {"time"})
    times = []
    clock_matches = list(re.finditer(r"\b(\d{1,2}):(\d{2})\s*(am|pm)?\b", text))
    for match in clock_matches:
        hour, minute, meridiem = int(match[1]), int(match[2]), match[3]
        if minute > 59 or hour > 23 or (meridiem and not 1 <= hour <= 12):
            raise Clarification("That time is invalid. Please use a valid HH:MM or AM/PM appointment time in IST.", {"time"})
        if meridiem:
            hour = hour % 12 + (12 if meridiem == "pm" else 0)
        times.append(f"{hour:02}:{minute:02}")
    without_clocks = re.sub(r"\b\d{1,2}:\d{2}\s*(?:am|pm)?\b", "", text)
    for match in re.finditer(r"\b(\d{1,2})\s*(am|pm)\b", without_clocks):
        hour = int(match[1])
        if not 1 <= hour <= 12:
            raise Clarification("That time is invalid. Please use a valid AM/PM time.", {"time"})
        times.append(f"{hour % 12 + (12 if match[2] == 'pm' else 0):02}:00")
    if len(set(times)) > 1:
        raise Clarification("I found more than one appointment time. Please choose one, or say ‘any time’.", {"time"})
    if times and periods and (int(times[0][:2]) >= 12) != ("afternoon" in periods):
        raise Clarification("The time and morning/afternoon request disagree. Which time should I use?", {"time"})
    if re.search(r"\b(?:no time preference|any time)\b", text):
        if times or periods:
            raise Clarification("Would you like the stated time, or any time?", {"time"})
        values.update(hard_period=None, soft_period=None, exact_time=None)
        supplied.add("time")
    elif times:
        values.update(exact_time=times[0], hard_period=None, soft_period=None)
        supplied.add("time")
    elif periods:
        period = next(iter(periods))
        soft = bool(re.search(r"\b(?:prefer|preferred|if possible|ideally)\b", text))
        hard = bool(re.search(r"\b(?:only|must|need to|have to)\b", text)) or not soft
        values.update(hard_period=period if hard else None, soft_period=None if hard else period, exact_time=None)
        supplied.add("time")
    elif re.search(r"\b(?:at|around)\s+\d+\b|\b(?:noon|midnight|evening|night)\b", text):
        raise Clarification("Please specify morning, afternoon, or an exact HH:MM or AM/PM time in IST.", {"time"})
    if not supplied and re.search(r"\b(?:actually|change|instead|make that|reschedule)\b", text):
        raise Clarification("I couldn’t understand that correction. Please restate the appointment date and time in IST before confirming.", {"day", "time"})
    return Request(**values), supplied
