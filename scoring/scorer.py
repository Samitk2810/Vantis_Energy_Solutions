from tools.service_area import check_service_area

# All tunable numbers live here. Retargeting to another client = edit this block.
WEIGHTS = {
    "intent":   {"quote": 25, "sizing": 15, "general": 5},
    "clarity":  {"kw_or_package": 20, "bill_and_property": 14, "partial": 7, "none": 0},
    "location": {"supported": 20, "unverified": 8, "none": 0},
    "timeline": {"within_1m": 20, "within_3m": 14, "within_6m": 8, "later": 2, "unknown": 0},
    "contact":  {"both": 10, "one": 6, "none": 0},
    "energy_info": 5,
}
HOT_MIN = 70
WARM_MIN = 40

ACTIONS = {
    "HOT": "Contact within 1 hour and book a site assessment.",
    "WARM": "Contact within 24 hours; confirm the missing details.",
    "COLD": "Add to nurture list; follow up if the customer returns.",
}


def _has(fields, key):
    return fields.get(key) not in (None, "", [])


def score_lead(fields: dict, location_status: str | None = None) -> dict:
    """Deterministic lead score. Returns score, status, reason, action and a breakdown."""
    # location
    city = fields.get("city")
    if not location_status and city:
        location_status, _ = check_service_area(city)
    loc_key = location_status if location_status in ("supported", "unverified") else "none"

    # intent
    intent = fields.get("intent") if fields.get("intent") in WEIGHTS["intent"] else "general"

    # requirement clarity
    has_size = _has(fields, "system_size_kw") or _has(fields, "package")
    has_energy = _has(fields, "monthly_bill") or _has(fields, "monthly_units")
    has_property = _has(fields, "property_type")
    if has_size:
        clarity = "kw_or_package"
    elif has_energy and has_property:
        clarity = "bill_and_property"
    elif has_energy or has_property:
        clarity = "partial"
    else:
        clarity = "none"

    # timeline
    months = fields.get("timeline_months")
    if months is None:
        tl = "unknown"
    elif months <= 1:
        tl = "within_1m"
    elif months <= 3:
        tl = "within_3m"
    elif months <= 6:
        tl = "within_6m"
    else:
        tl = "later"

    # contact
    n_contact = int(_has(fields, "phone")) + int(_has(fields, "email"))
    contact = "both" if n_contact == 2 else "one" if n_contact == 1 else "none"

    breakdown = {
        "intent": WEIGHTS["intent"][intent],
        "requirement_clarity": WEIGHTS["clarity"][clarity],
        "location": WEIGHTS["location"][loc_key],
        "timeline": WEIGHTS["timeline"][tl],
        "contact": WEIGHTS["contact"][contact],
        "energy_info": WEIGHTS["energy_info"] if has_energy else 0,
    }
    score = sum(breakdown.values())

    # status
    if score >= HOT_MIN and loc_key == "supported" and n_contact >= 1:
        status = "HOT"
    elif score >= WARM_MIN or (intent == "quote" and n_contact >= 1):
        status = "WARM"      # also: asked for a quote/callback and left contact details
    else:
        status = "COLD"

    action = ACTIONS[status]
    if loc_key == "unverified":
        action = "Verify service availability for the location, then contact the customer."
    elif status != "COLD" and n_contact == 0:
        action = "Collect a phone number or email before follow-up."

    return {
        "lead_score": score,
        "lead_status": status,
        "reason": _reason(fields, intent, clarity, loc_key, tl, contact, has_energy),
        "recommended_action": action,
        "score_breakdown": breakdown,
    }


def _reason(fields, intent, clarity, loc_key, tl, contact, has_energy) -> str:
    got, missing = [], []
    got.append({"quote": "Asked for a quote or installation",
                "sizing": "Asked about sizing or cost",
                "general": "General enquiry"}[intent])
    if clarity == "kw_or_package":
        got.append("specific system requirement")
    elif clarity in ("bill_and_property", "partial"):
        got.append("partial requirement details")
    else:
        missing.append("requirement details")
    city = fields.get("city")
    if loc_key == "supported":
        got.append(f"{city} is a serviceable location")
    elif loc_key == "unverified":
        got.append(f"{city} is not yet verified for service")
    else:
        missing.append("location")
    if tl in ("within_1m", "within_3m", "within_6m"):
        got.append("timeline given")
    elif tl == "later":
        got.append("timeline is more than 6 months away")
    else:
        missing.append("timeline")
    if contact == "both":
        got.append("phone and email provided")
    elif contact == "one":
        got.append("one contact method provided")
    else:
        missing.append("contact details")
    text = "; ".join(got) + "."
    if missing:
        text += " Missing: " + ", ".join(missing) + "."
    return text