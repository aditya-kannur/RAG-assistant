SCHEME_NAMES = [
    "PM-KISAN",
    "Ayushman Bharat (PM-JAY)",
    "PMAY-Urban 2.0",
    "PM Kisan Maandhan Yojana (PM-KMY)",
]

# Simple alias map — catches common ways users might refer to a scheme
SCHEME_ALIASES = {
    "pm-kisan": "PM-KISAN",
    "pm kisan": "PM-KISAN",
    "kisan samman nidhi": "PM-KISAN",
    "ayushman bharat": "Ayushman Bharat (PM-JAY)",
    "pm-jay": "Ayushman Bharat (PM-JAY)",
    "pmjay": "Ayushman Bharat (PM-JAY)",
    "pmay": "PMAY-Urban 2.0",
    "pradhan mantri awas yojana": "PMAY-Urban 2.0",
    "pm-kmy": "PM Kisan Maandhan Yojana (PM-KMY)",
    "kisan maandhan": "PM Kisan Maandhan Yojana (PM-KMY)",
}


def extract_scheme_filter(question: str) -> str | None:
    """
    If the user's question explicitly names a scheme, return its canonical
    name so retrieval can be filtered to just that scheme's chunks.
    Returns None if no scheme is clearly named (search stays unfiltered).
    """
    q_lower = question.lower()
    for alias, canonical_name in SCHEME_ALIASES.items():
        if alias in q_lower:
            return canonical_name
    return None