"""Arabic Text Normalizer — تطبيع النصوص العربية للبحث الدلالي

يُحسن دقة البحث العربي عن طريق تطبيع الاختلافات الإملائية:
- إزالة الحركات (fat7a, damma, kasra, sukun, shadda...)
- توحيد الهمزات (أ/إ/آ → ا، ؤ → و، ئ → ي)
- تطبيع الياء والألف المقصورة (ى → ي)
- تطبيع الحروف الأعجمية (ک → ك، ی → ي)
- إزالة التشكيل الزائد

ملاحظة: لا نزيل "الـ" التعريف أو حروف الجر الملتصقة
لأنها قد تحمل معنى دلالي مهم في البحث.
"""

import re


# أنماط الحركات العربية Unicode
_DIACRITICS_PATTERN = re.compile(r'[\u064B-\u065F\u0670\u0640]')

# أنماط الهمزات على الألف
_ALEF_PATTERN = re.compile(r'[آأإ]')

# أنماط الهمزات على الواو والياء
_HAMZA_WAW_PATTERN = re.compile(r'ؤ')
_HAMZA_YAA_PATTERN = re.compile(r'ئ')
_HAMZA_LONE_PATTERN = re.compile(r'ء')

# أنماط الحروف الأعجمية
_PERSIAN_KAF_PATTERN = re.compile(r'ک')  # U+06A9
_PERSIAN_YAA_PATTERN = re.compile(r'ی')  # U+06CC

# الألف المقصورة
_ALEF_MAKSURA_PATTERN = re.compile(r'ى')

# أنماط التاء المربوطة والهاء
_TA_MARBUTA_PATTERN = re.compile(r'ة')


def remove_diacritics(text: str) -> str:
    """إزالة الحركات العربية (fat7a, damma, kasra, sukun, shadda, tatweel)

    Examples:
        >>> remove_diacritics('الذَّكاءُ الاصْطِناعي')
        'الذكاء الاصطناعي'
        >>> remove_diacritics('بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ')
        'بسم الله الرحمن الرحيم'
    """
    return _DIACRITICS_PATTERN.sub('', text)


def normalize_alef(text: str) -> str:
    """توحيد الهمزات: آ/أ/إ → ا

    Examples:
        >>> normalize_alef('إسلام')
        'اسلام'
        >>> normalize_alef('آمنوا')
        'امنوا'
        >>> normalize_alef('أحد')
        'احد'
    """
    return _ALEF_PATTERN.sub('ا', text)


def normalize_hamza_extended(text: str) -> str:
    """تطبيع الهمزات على الواو والياء، والحروف الأعجمية

    Examples:
        >>> normalize_hamza_extended('مؤسسة')
        'موسسة'
        >>> normalize_hamza_extended('سؤال')
        'سوال'
        >>> normalize_hamza_extended('مسؤول')
        'مسوول'
        >>> normalize_hamza_extended('قارئ')
        'قاري'
    """
    text = _HAMZA_WAW_PATTERN.sub('و', text)
    text = _HAMZA_YAA_PATTERN.sub('ي', text)
    text = _HAMZA_LONE_PATTERN.sub('', text)
    text = _PERSIAN_KAF_PATTERN.sub('ك', text)
    text = _PERSIAN_YAA_PATTERN.sub('ي', text)
    return text


def normalize_yaa(text: str) -> str:
    """توحيد الياء والألف المقصورة: ى → ي

    Examples:
        >>> normalize_yaa('هداية')
        'هداية'
        >>> normalize_yaa('موسى')
        'موسي'
        >>> normalize_yaa('مستشفى')
        'مستشفي'
    """
    return _ALEF_MAKSURA_PATTERN.sub('ي', text)


def normalize_ta_marbuta(text: str) -> str:
    """تحويل التاء المربوطة إلى هاء: ة → ه

    Examples:
        >>> normalize_ta_marbuta('اللغة العربية')
        'اللغه العربيه'
    """
    return _TA_MARBUTA_PATTERN.sub('ه', text)


def normalize_whitespace(text: str) -> str:
    """تطبيع المسافات الزائدة

    Examples:
        >>> normalize_whitespace('  hello   world  ')
        'hello world'
    """
    return re.sub(r'\s+', ' ', text).strip()


def normalize_query(text: str) -> str:
    """تطبيع كامل لاستعلام عربي

    يطبق جميع عمليات التطبيع المناسبة للبحث الدلالي:
    1. إزالة الحركات
    2. توحيد الهمزات (أ/إ/آ + ؤ/ئ/ء)
    3. تطبيع الياء والألف المقصورة
    4. تطبيع الحروف الأعجمية
    5. تطبيع المسافات

    لا يزيل:
    - "الـ" التعريف (قد يغير المعنى)
    - حروف الجر الملتصقة (قد تحمل معنى)
    - التاء المربوطة (قد تغير قراءة الكلمة)

    Args:
        text: النص المراد تطبيعه

    Returns:
        النص بعد التطبيع

    Examples:
        >>> normalize_query('الْكِتَابُ')
        'الكتاب'
        >>> normalize_query('إِسْلَام')
        'اسلام'
        >>> normalize_query('آمَنُوا')
        'امنوا'
        >>> normalize_query('مؤسسة')
        'موسسة'
        >>> normalize_query('مستشفى الى')
        'مستشفي الي'
    """
    if not text:
        return text

    # 1. إزالة الحركات
    text = remove_diacritics(text)

    # 2. توحيد الهمزات
    text = normalize_alef(text)

    # 3. الهمزات الممتدة (ؤ/ئ/ء) + الحروف الأعجمية
    text = normalize_hamza_extended(text)

    # 4. الألف المقصورة ى → ي
    text = normalize_yaa(text)

    # 5. تطبيع المسافات
    text = normalize_whitespace(text)

    return text


def is_arabic(text: str) -> bool:
    """كشف النص العربي

    Args:
        text: النص المراد فحصه

    Returns:
        True إذا كان النص يحتوي على أحرف عربية
    """
    return bool(re.search(r'[\u0600-\u06FF]', text))


def get_arabic_ratio(text: str) -> float:
    """نسبة الأحرف العربية في النص

    Args:
        text: النص المراد فحصه

    Returns:
        نسبة الأحرف العربية (0.0 إلى 1.0)
    """
    if not text:
        return 0.0

    arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
    total_chars = len(re.findall(r'\S', text))

    if total_chars == 0:
        return 0.0

    return arabic_chars / total_chars
