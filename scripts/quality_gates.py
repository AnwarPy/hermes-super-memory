#!/usr/bin/env python3
"""
Quality gates for Hermes Memory — single source of truth.

Import from anywhere:
    from quality_gates import is_junk_fact, is_sensitive_fact, normalize_arabic_text
"""

import re

# ============================================================
# Arabic normalization (canonical — used by ALL writers)
# ============================================================

_ARABIC_DIACRITICS = re.compile(r'[\u064B-\u065F\u0610-\u061A\u0670]')
_ARABIC_ALEF = re.compile(r'[إأآٱ]')
_ARABIC_YAA = re.compile(r'ى')
_ARABIC_TA_MARBUTA = re.compile(r'ة')


def normalize_arabic_text(text: str) -> str:
    """Canonical Arabic normalization — the ONLY normalizer to use."""
    if not text:
        return text
    text = _ARABIC_DIACRITICS.sub('', text)
    text = _ARABIC_ALEF.sub('ا', text)
    text = _ARABIC_YAA.sub('ي', text)
    text = _ARABIC_TA_MARBUTA.sub('ه', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def is_arabic_heavy(text: str, threshold: float = 0.3) -> bool:
    """Check if text is predominantly Arabic."""
    if not text:
        return False
    arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
    total_chars = len(re.findall(r'\S', text))
    return total_chars > 0 and (arabic_chars / total_chars) >= threshold


# ============================================================
# Skip phrases (single source of truth)
# ============================================================

SKIP_PHRASES = [
    "help was provided", "assistant helped", "conversation was about",
    "system was built", "all dependencies", "instructions were sent",
    "no technical details", "no specific", "no explicit",
    "the user asked", "user asked a", "the conversation was",
    "the assistant was", "no code or technical",
    "المحادثة كانت حول", "تم تقديم المساعدة", "تم التحقق من",
    "تم بناء النظام", "تم تثبيت جميع", "تم ارسال تعليمات",
    "لا توجد تفاصيل", "لا يوجد", "لم يتم",
]

MIN_FACT_KEY_LENGTH_AR = 10
MIN_FACT_KEY_LENGTH_EN = 20

# P0-1: Security gate
_SENSITIVE_PATTERN = re.compile(r'\b[A-Z_]{2,40}(?:KEY|TOKEN|SECRET|PASSWORD|API_KEY|BOT_TOKEN)\b')


def is_sensitive_fact(key: str) -> bool:
    """Security gate: reject facts that expose API keys, tokens, or secrets."""
    if not key:
        return False
    key_stripped = key.strip()
    if re.match(r'^[A-Z_]{3,}$', key_stripped):
        return True
    if _SENSITIVE_PATTERN.search(key_stripped) and len(key_stripped.split()) <= 5:
        return True
    return False


# ============================================================
# P4: Snapshot-stat keyword list
# ============================================================

_STAT_KEYWORDS = frozenset([
    'nodes', 'edges', 'vertices', 'modularity', 'clusters', 'communities',
    'components', 'degree', 'density', 'diameter', 'centrality',
    'lines', 'bytes', 'files', 'directories', 'functions', 'classes',
    'parameters', 'variables', 'tokens', 'words', 'characters',
    'records', 'rows', 'columns', 'entries', 'items',
    'sessions', 'messages', 'connections', 'requests', 'responses',
    # Singular forms (LLMs mix singular/plural)
    'node', 'edge', 'vertex', 'cluster', 'community',
    'component', 'line', 'file', 'directory', 'function', 'class',
    'record', 'row', 'column', 'entry', 'item',
    'session', 'message', 'connection', 'request', 'response',
    'key', 'keys', 'fact', 'facts',
    # Arabic
    'عقد', 'حواف', 'رؤوس', 'وحدات', 'مجموعات', 'مكونات',
    'أسطر', 'بايت', 'ملفات', 'مجلدات', 'دوال', 'أصناف',
    'رموز', 'كلمات', 'أحرف', 'سجلات', 'صفوف', 'أعمدة',
    'جلسات', 'رسائل', 'اتصالات', 'طلبات',
    'عقدة', 'حافة', 'ملف', 'مجلد', 'دالة', 'صنف',
    'رمز', 'كلمة', 'حرف', 'سجل', 'صف', 'عمود',
    'جلسة', 'رسالة', 'اتصال', 'طلب', 'مفتاح', 'مفاتيح', 'حقيقة', 'حقائق',
])

_STAT_KEYWORDS_RE = '|'.join(_STAT_KEYWORDS)

# Pre-compiled P4 patterns (English — run on key_lower)
_P4_EN_PATTERNS = [
    # "number of X is N", "total X: N" — \s* before separator for "total X: N"
    re.compile(r'(?:number\s+of|total|count\s+of|average|mean)\s+.+?\s*(?:is|:|=|was)\s*[\d,.]+', re.IGNORECASE),
    # Pattern 1b: "X dimension is N", "X distribution is N"
    # Excludes "rate", "level" to avoid false positives on API limits etc.
    re.compile(r'\b\w+\s+(?:score|value|index|dimension|count|total|amount|size|number|percentage|distribution)\s+(?:is|:|=|was)\s*[\d,.]+', re.IGNORECASE),
    # "N nodes", "492 edges", "1 session" — digit(s) then stat keyword
    # Use \\d+(?:[.,]\\d+)* to match real numbers only (123, 1.5, 1,234)
    # Avoid matching bare commas like ", edges," in normal sentences
    re.compile(r'\d+(?:[.,]\d+)*\s+(?:' + _STAT_KEYWORDS_RE + r')(?:\s|,|$)', re.IGNORECASE),
    # "modularity is N", "nodes: N", "sessions in"
    re.compile(r'\b(?:' + _STAT_KEYWORDS_RE + r')\s+(?:is|:|=|was|of|has|in|are)\s*[\d,.]+', re.IGNORECASE),
    # Any percentage
    re.compile(r'[\d,.]+\s*%'),
]

# Pre-compiled P4 patterns (Arabic — run on key_normalized)
_P4_AR_PATTERNS = [
    re.compile(r'(?:عدد|اجمالي|مجموع|متوسط)\s+\S+?\s*[:=]\s*[\d,.]+'),
    re.compile(r'(?:عدد|اجمالي|مجموع|متوسط)\s+\S+?\s+(?:هو|يساوي)\s*[\d,.]+'),
]


def is_junk_fact(key: str) -> bool:
    """Quality gate: reject junk/placeholder/snapshot-stat facts.

    Returns True if the fact should be SKIPPED.
    """
    if not key:
        return True

    key = key.strip()
    if not key:
        return True

    # P0-1: Security gate
    if is_sensitive_fact(key):
        return True

    # Length check
    if is_arabic_heavy(key):
        if len(key) < MIN_FACT_KEY_LENGTH_AR:
            return True
    else:
        if len(key) < MIN_FACT_KEY_LENGTH_EN:
            return True

    # Skip phrase matching
    key_normalized = normalize_arabic_text(key.lower())
    for phrase in SKIP_PHRASES:
        phrase_normalized = normalize_arabic_text(phrase.lower())
        if phrase_normalized in key_normalized:
            return True

    # Pure stats: "13 nodes"
    if re.match(r'^\d+\s+\w+$', key):
        return True

    # Passive Arabic: "تم التحقق"
    if re.match(r'^تم\s+\w+\s*$', key):
        return True

    # snake_case
    if re.match(r'^[a-z][a-z0-9_]{15,}$', key) and not is_arabic_heavy(key) and ' ' not in key:
        return True

    # P4: Snapshot statistics — English patterns on lowercased text
    key_lower = key.lower()
    for pat in _P4_EN_PATTERNS:
        if pat.search(key_lower):
            return True

    # P4: Arabic patterns on normalized text (إ→ا, ة→ه)
    for pat in _P4_AR_PATTERNS:
        if pat.search(key_normalized):
            return True

    # P4 fallback: >30% digits/punct + stat keyword
    # Raised from 0.20 to 0.30 to avoid false positives on version/port facts
    # like "Node.js 22, Docker, PostgreSQL 17"
    non_space = key.replace(' ', '')
    if non_space and sum(1 for c in non_space if c.isdigit() or c in '.,') / len(non_space) > 0.30:
        key_words = set(key_lower.split())
        if any(kw in key_words for kw in _STAT_KEYWORDS):
            return True

    return False
