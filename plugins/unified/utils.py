"""Utility functions for Unified Memory Provider."""

import re


def _format_age(seconds):
    """P1: تنسيق عمر النتيجة للعرض."""
    if seconds < 60:
        return "now"
    if seconds < 3600:
        return f"{int(seconds/60)}m ago"
    if seconds < 86400:
        return f"{int(seconds/3600)}h ago"
    return f"{int(seconds/86400)}d ago"


def _clean_chunk(content, max_len=300):
    """تنظيف بسيط للنصوص — بديل 10 سطور لـ 165 سطر regex هش.
    
    P1: الحقائق من LLM نظيفة أصلاً. هذا يكفي.
    FTS5 snippets تستخدم نفس الدالة.
    حد أدنى 15 حرف للمحتوى المفيد.
    """
    if not content:
        return ""
    # إزالة أحرف التحكم وعلامات FTS5
    content = content.replace('>>>', '').replace('<<<', '')
    content = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', content)
    content = content.replace('\\r\\n', ' ').replace('\\n', ' ').replace('\\t', ' ')
    # تطبيع مسافات واقتطاع
    content = re.sub(r'\s+', ' ', content).strip()
    if len(content) > max_len:
        content = content[:max_len].rsplit(' ', 1)[0] + '...'
    return content if len(content) >= 15 else ""
