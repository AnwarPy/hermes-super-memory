#!/usr/bin/env python3
"""
Quality gates for Hermes Memory — single source of truth.

P0 fix: Unify SKIP_PHRASES, MIN_FACT_KEY_LENGTH, and Arabic normalization
across session_summarizer, fact_extractor, and graph_updater.

Import from anywhere:
    from quality_gates import is_junk_fact, normalize_fact_key, MIN_FACT_KEY_LENGTH
"""

import re

# ============================================================
# Arabic normalization (canonical — used by ALL writers)
# ============================================================

# P0-1: Unified normalization includes ta-marbuta (ة → ه)
# This fixes silent duplicate nodes: نفس الحقيقة بصيغتين مختلفتين
# كانت تدخل BGE-M3 بشكل مختلف → embeddings مختلفة → dedup يفشل

_ARABIC_DIACRITICS = re.compile(r'[\u064B-\u065F\u0610-\u061A\u0670]')
_ARABIC_ALEF = re.compile(r'[إأآٱ]')
_ARABIC_YAA = re.compile(r'ى')
_ARABIC_TA_MARBUTA = re.compile(r'ة')
_ARABIC_FOREIGN = re.compile(r'[^\u0600-\u06FFa-zA-Z0-9\s.,;:?!()\-_/@#$%&*+=]')


def normalize_arabic_text(text: str) -> str:
    """Canonical Arabic normalization — the ONLY normalizer to use.
    
    Handles: diacritics, alef variants, ya/ta marbuta, foreign chars, whitespace.
    Used for: fact hashing, embedding input, skip phrase matching, dedup keys.
    """
    if not text:
        return text
    
    # 1. Remove diacritics
    text = _ARABIC_DIACRITICS.sub('', text)
    
    # 2. Unify alef variants
    text = _ARABIC_ALEF.sub('ا', text)
    
    # 3. Unify ya variants (including maqsura)
    text = _ARABIC_YAA.sub('ي', text)
    
    # 4. Unify ta-marbuta to ha (P0-1: was inconsistent across writers)
    text = _ARABIC_TA_MARBUTA.sub('ه', text)
    
    # 5. Normalize whitespace
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
    # English
    "help was provided", "assistant helped", "conversation was about",
    "system was built", "all dependencies", "instructions were sent",
    "no technical details", "no specific", "no explicit",
    "the user asked", "user asked a", "the conversation was",
    "the assistant was", "no code or technical",
    # Arabic (will be normalized before matching)
    "المحادثة كانت حول", "تم تقديم المساعدة", "تم التحقق من",
    "تم بناء النظام", "تم تثبيت جميع", "تم ارسال تعليمات",
    "لا توجد تفاصيل", "لا يوجد", "لم يتم",
]

# P0-1: Arabic-aware minimum length
# Arabic encodes more meaning per character
MIN_FACT_KEY_LENGTH_AR = 10
MIN_FACT_KEY_LENGTH_EN = 20


def is_junk_fact(key: str) -> bool:
    """Quality gate: reject junk/placeholder facts.
    
    P0-5: Single gate used by all writers (summarizer, extractor, graph_updater).
    P0-1: Uses unified Arabic normalization + language-aware length check.
    
    Returns True if the fact should be SKIPPED.
    """
    if not key:
        return True
    
    key = key.strip()
    if not key:
        return True
    
    key_lower = key.lower()
    key_normalized = normalize_arabic_text(key_lower)
    
    # Language-aware length check (P0-1)
    if is_arabic_heavy(key):
        if len(key) < MIN_FACT_KEY_LENGTH_AR:
            return True
    else:
        if len(key) < MIN_FACT_KEY_LENGTH_EN:
            return True
    
    # Skip phrase matching (with normalization)
    for phrase in SKIP_PHRASES:
        phrase_normalized = normalize_arabic_text(phrase.lower())
        if phrase_normalized in key_normalized:
            return True
    
    # Reject facts that are pure stats without context
    # e.g. "13 nodes", "492 lines", "8080 port"
    if re.match(r'^\d+\s+\w+$', key.strip()):
        return True
    
    # Reject passive-voice Arabic with no object
    # e.g. "تم التحقق", "تم الفحص", "تم الانتهاء"
    if re.match(r'^تم\s+\w+\s*$', key.strip()):
        return True
    
    return False
