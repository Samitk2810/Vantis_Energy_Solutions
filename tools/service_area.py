import re
from typing import Optional

SERVICE_AREAS = {
    "Uttar Pradesh": ["Lucknow", "Prayagraj", "Kanpur", "Varanasi", "Noida", "Ghaziabad"],
    "Delhi NCR": ["Delhi", "Gurugram", "Noida", "Ghaziabad", "Faridabad"],
    "Punjab": ["Ludhiana", "Amritsar", "Jalandhar"],
    "Haryana": ["Panipat", "Karnal", "Sonipat"],
    "Rajasthan": ["Jaipur", "Kota"],
}

# Alternate names for listed cities
ALIASES = {
    "gurgaon": "Gurugram",
    "allahabad": "Prayagraj",
    "banaras": "Varanasi",
    "benares": "Varanasi",
    "new delhi": "Delhi",
}

# Places that contain a listed city's name but are NOT listed
NOT_LISTED_PHRASES = ["greater noida"]

SUPPORTED = {c.lower(): c for cities in SERVICE_AREAS.values() for c in cities}
_NAMES = sorted(list(SUPPORTED) + list(ALIASES), key=len, reverse=True)


def find_supported_city(text: str) -> Optional[str]:
    """Return a listed city mentioned in the text, or None. No LLM involved."""
    t = text.lower()
    for phrase in NOT_LISTED_PHRASES:
        t = t.replace(phrase, " ")
    for name in _NAMES:
        if re.search(rf"\b{re.escape(name)}\b", t):
            return SUPPORTED.get(name) or ALIASES[name]
    return None


def check_service_area(city: str) -> tuple[str, str]:
    """Return (status, city). status is 'supported' or 'unverified'."""
    key = city.strip().lower()
    if key in SUPPORTED:
        return "supported", SUPPORTED[key]
    if key in ALIASES:
        return "supported", ALIASES[key]
    return "unverified", city.strip().title()