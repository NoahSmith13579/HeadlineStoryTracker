"""Headline cleaning implementation stub for Phase 1."""
import html
import re
import unicodedata
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from langdetect import DetectorFactory, LangDetectException, detect

from hst.config import SEED

DetectorFactory.seed = SEED

TRACKING_PARAMS = {
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "ref",
}


def clean_title(title: str) -> str:
    text = html.unescape(title)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+\|\s*[^|]{1,40}\s*$", "", text)
    if not re.match(r"^(LIVE|WATCH|UPDATE)\b", text, flags=re.IGNORECASE):
        text = re.sub(r"\s+-\s+(?:The\s+)?[A-Za-z][A-Za-z0-9 .&'-]{1,40}\s*$", "", text)
        text = re.sub(r"\s+[—–]\s*[A-Za-z][A-Za-z0-9 .&'-]{1,40}\s*$", "", text)
    text = re.sub(r"^(LIVE|WATCH|UPDATE)\s*:\s*", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def canonicalize_url(url: str) -> str:
    parsed = urlsplit(url.strip())
    host = (parsed.hostname or "").lower()
    if parsed.port:
        netloc = f"{host}:{parsed.port}"
    else:
        netloc = host
    if parsed.username:
        netloc = f"{parsed.username}@{netloc}"
    query = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key.lower() not in TRACKING_PARAMS and not key.lower().startswith("utm_")
    ]
    return urlunsplit((parsed.scheme.lower(), netloc, parsed.path or "/", urlencode(query), ""))


def detect_language(text: str) -> str:
    if not text or len(text.strip()) < 3:
        return "und"
    try:
        return detect(text)
    except LangDetectException:
        return "und"


def normalize_title(title: str) -> str:
    text = html.unescape(title)
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()
