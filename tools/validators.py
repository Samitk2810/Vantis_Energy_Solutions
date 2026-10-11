import re
from typing import Optional

EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")


def clean_phone(raw: Optional[str]) -> Optional[str]:
    """Return an Indian mobile number as +91XXXXXXXXXX, or None if invalid."""
    if not raw:
        return None
    digits = re.sub(r"\D", "", str(raw))
    if len(digits) == 12 and digits.startswith("91"):
        digits = digits[2:]
    elif len(digits) == 11 and digits.startswith("0"):
        digits = digits[1:]
    if re.fullmatch(r"[6-9]\d{9}", digits):
        return f"+91{digits}"
    return None


def clean_email(raw: Optional[str]) -> Optional[str]:
    """Return a lower-cased email, or None if invalid."""
    if not raw:
        return None
    email = str(raw).strip().lower()
    return email if EMAIL_RE.match(email) else None