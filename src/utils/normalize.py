"""
Shared normalization helpers for business names and addresses.
Used by both the blocking stage and the feature-engineering stage,
so keep these two use-cases in mind when editing.
"""

import re

# Common legal-suffix variants -> canonical short form
LEGAL_SUFFIX_MAP = {
    "corporation": "corp",
    "incorporated": "inc",
    "limited": "ltd",
    "private": "pvt",
    "company": "co",
    "and": "&",
}

# Common address abbreviation variants -> canonical short form
ADDRESS_ABBREV_MAP = {
    "road": "rd",
    "street": "st",
    "avenue": "ave",
    "boulevard": "blvd",
    "drive": "dr",
    "lane": "ln",
    "apartment": "apt",
    "building": "bldg",
    "floor": "fl",
}

_PUNCT_RE = re.compile(r"[^\w\s]")
_MULTISPACE_RE = re.compile(r"\s+")


def _clean_base(text: str) -> str:
    if text is None or (isinstance(text, float)):
        return ""
    text = str(text).lower().strip()
    text = _PUNCT_RE.sub(" ", text)
    text = _MULTISPACE_RE.sub(" ", text).strip()
    return text


def normalize_name(name: str) -> str:
    """Lowercase, strip punctuation, canonicalize legal suffixes."""
    text = _clean_base(name)
    tokens = text.split()
    tokens = [LEGAL_SUFFIX_MAP.get(tok, tok) for tok in tokens]
    return " ".join(tokens)


def normalize_address(address: str) -> str:
    """Lowercase, strip punctuation, canonicalize common address abbreviations."""
    text = _clean_base(address)
    tokens = text.split()
    tokens = [ADDRESS_ABBREV_MAP.get(tok, tok) for tok in tokens]
    return " ".join(tokens)


def name_tokens(name: str) -> set:
    """Token set of a normalized name, useful for Jaccard / blocking keys."""
    return set(normalize_name(name).split())


def address_tokens(address: str) -> set:
    return set(normalize_address(address).split())


def blocking_key(name: str) -> str:
    """
    A coarse blocking key: sorted first-3-token signature of the normalized
    name. Cheap and works reasonably across languages/transliterations
    since it doesn't depend on exact spelling beyond tokenization.
    """
    tokens = sorted(name_tokens(name))
    return " ".join(tokens[:3])
