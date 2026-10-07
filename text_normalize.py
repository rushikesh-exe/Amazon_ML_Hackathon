"""
Text normalization for business_name / business_address fields.

Purpose: reduce the noise types called out in the problem statement
(abbreviations, legal suffixes, transliteration, punctuation) BEFORE
embedding, so semantically identical strings land closer in vector space.
"""
import re
import unicodedata
from unidecode import unidecode

# Word-boundary abbreviation expansions. Keys/values are lowercase;
# applied after lowercasing the full string.
ABBREVIATIONS = {
    r"\brd\b": "road",
    r"\bst\b": "street",
    r"\bave\b": "avenue",
    r"\bdr\b": "drive",
    r"\bln\b": "lane",
    r"\bblvd\b": "boulevard",
    r"\bapt\b": "apartment",
    r"\bste\b": "suite",
    r"\bunit\b": "unit",  # kept explicit, no-op, documents intent
    r"\bpvt\b": "private",
    r"\bltd\b": "limited",
    r"\bcorp\b": "corporation",
    r"\binc\b": "incorporated",
    r"\bco\b": "company",
    r"\bllc\b": "llc",  # no expansion, common enough to leave as-is
    r"\bpo box\b": "po box",
    r"&": " and ",
}

_ABBREV_PATTERNS = [(re.compile(pat), repl) for pat, repl in ABBREVIATIONS.items()]

# Keep letters, digits, spaces, and apostrophes (e.g. "Orelee's").
_STRIP_CHARS_RE = re.compile(r"[^a-z0-9'\s]")
_MULTI_SPACE_RE = re.compile(r"\s+")


def normalize_text(text) -> str:
    """Normalize a single field (business_name or business_address).

    Steps: NaN -> "", transliterate non-Latin scripts to ASCII
    (Devanagari, etc.), lowercase, expand abbreviations, strip
    punctuation (except apostrophes), collapse whitespace.
    """
    if text is None or (isinstance(text, float) and text != text):  # NaN check
        return ""
    text = str(text).strip()
    if not text:
        return ""

    # Transliterate non-Latin scripts (e.g. Devanagari) to a Latin
    # approximation so cross-script name variants have a chance to match.
    text = unidecode(text)

    text = text.lower()

    for pattern, repl in _ABBREV_PATTERNS:
        text = pattern.sub(repl, text)

    text = _STRIP_CHARS_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def build_row_text(business_name, business_address) -> str:
    """Concatenate normalized name + address into one string for embedding."""
    name = normalize_text(business_name)
    addr = normalize_text(business_address)
    return f"{name} {addr}".strip()


if __name__ == "__main__":
    # quick smoke test
    samples = [
        ("Orelee's Barbershop", "1795 Westchester Drive, High Point, NC"),
        ("राम मार्केटिंग प्राइवेट लिमिटेड", "KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi"),
        ("B+ Retail Inc", "1712 Montebello Avenue, Phoenix, AZ"),
    ]
    for name, addr in samples:
        print(build_row_text(name, addr))
