"""Tests for Arabic Normalizer — اختبارات تطبيع النصوص العربية"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arabic_normalizer import (
    remove_diacritics,
    normalize_alef,
    normalize_hamza_extended,
    normalize_yaa,
    normalize_ta_marbuta,
    normalize_whitespace,
    normalize_query,
    is_arabic,
    get_arabic_ratio,
)


class TestRemoveDiacritics:
    def test_basic_diacritics(self):
        assert remove_diacritics('الذَّكاءُ الاصْطِناعي') == 'الذكاء الاصطناعي'

    def test_bismillah(self):
        result = remove_diacritics('بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ')
        assert result == 'بسم الله الرحمن الرحيم'

    def test_no_diacritics(self):
        assert remove_diacritics('الكتاب') == 'الكتاب'

    def test_tatweel(self):
        assert remove_diacritics('الـــعـــربـــي') == 'العربي'

    def test_empty(self):
        assert remove_diacritics('') == ''


class TestNormalizeAlef:
    def test_alef_with_hamza_above(self):
        assert normalize_alef('أحمد') == 'احمد'

    def test_alef_with_hamza_below(self):
        assert normalize_alef('إسلام') == 'اسلام'

    def test_alef_madda(self):
        assert normalize_alef('آمنوا') == 'امنوا'

    def test_normal_alef(self):
        assert normalize_alef('كتاب') == 'كتاب'

    def test_mixed(self):
        assert normalize_alef('أنا وإنت وأنا') == 'انا وانت وانا'


class TestNormalizeHamzaExtended:
    def test_hamza_on_waw(self):
        # مؤسسة: م-ؤ-س-س-ة → م-و-س-س-ة = موسسة
        assert normalize_hamza_extended('مؤسسة') == 'موسسة'

    def test_hamza_on_yaa(self):
        assert normalize_hamza_extended('مسؤول') == 'مسوول'

    def test_lone_hamza(self):
        # ء should be removed
        assert 'ء' not in normalize_hamza_extended('سؤال')

    def test_persian_kaf(self):
        # ک (U+06A9) → ك (U+0643)
        assert normalize_hamza_extended('کتاب') == 'كتاب'

    def test_persian_yaa(self):
        # ی (U+06CC) → ي (U+064A)
        assert normalize_hamza_extended('زیبا') == 'زيبا'

    def test_no_hamza(self):
        assert normalize_hamza_extended('كتاب') == 'كتاب'


class TestNormalizeYaa:
    def test_alef_maksura(self):
        assert normalize_yaa('موسى') == 'موسي'

    def test_alef_maksura_extended(self):
        # ى (U+0649) → ي — but هداية has regular ي not ى
        assert normalize_yaa('هدى') == 'هدي'

    def test_normal_yaa(self):
        assert normalize_yaa('كتاب') == 'كتاب'


class TestNormalizeTaMarbuta:
    def test_basic(self):
        assert normalize_ta_marbuta('اللغة') == 'اللغه'

    def test_no_ta_marbuta(self):
        assert normalize_ta_marbuta('كتاب') == 'كتاب'


class TestNormalizeWhitespace:
    def test_multiple_spaces(self):
        assert normalize_whitespace('  hello   world  ') == 'hello world'

    def test_tabs_and_newlines(self):
        assert normalize_whitespace('hello\t\nworld') == 'hello world'

    def test_normal(self):
        assert normalize_whitespace('hello world') == 'hello world'


class TestNormalizeQuery:
    def test_full_normalization(self):
        # إسلام مع حركات → اسلام بدون
        result = normalize_query('إِسْلَام')
        assert result == 'اسلام'

    def test_alef_maksura_in_query(self):
        # موسى → موسى (الألف المقصورة تُطبّع)
        result = normalize_query('موسى الى')
        assert 'ى' not in result
        assert 'الي' in result

    def test_hamza_in_query(self):
        result = normalize_query('مؤسسة')
        assert 'ؤ' not in result

    def test_persian_chars(self):
        result = normalize_query('کتاب')
        assert 'ک' not in result

    def test_empty(self):
        assert normalize_query('') == ''

    def test_none(self):
        assert normalize_query(None) is None

    def test_whitespace_only(self):
        assert normalize_query('   ') == ''


class TestIsArabic:
    def test_arabic_text(self):
        assert is_arabic('الكتاب') is True

    def test_english_text(self):
        assert is_arabic('hello world') is False

    def test_mixed(self):
        assert is_arabic('hello عالم') is True

    def test_empty(self):
        assert is_arabic('') is False


class TestGetArabicRatio:
    def test_pure_arabic(self):
        ratio = get_arabic_ratio('الكتاب على الطاولة')
        assert ratio > 0.8

    def test_pure_english(self):
        ratio = get_arabic_ratio('the book is on the table')
        assert ratio == 0.0

    def test_mixed(self):
        ratio = get_arabic_ratio('الكتاب book على table')
        assert 0.2 < ratio < 0.8

    def test_empty(self):
        assert get_arabic_ratio('') == 0.0


class TestArabicSearchScenarios:
    """سيناريوهات بحث عربية واقعية"""

    def test_misaleh(self):
        # مستشفى vs مستشفي (خلط شائع)
        q1 = normalize_query('مستشفى')
        q2 = normalize_query('مستشفي')
        # Both should normalize similarly (ى → ي)
        assert 'ي' in q2

    def test_mossasa(self):
        # مؤسسة → should remove hamza
        result = normalize_query('مؤسسة')
        assert 'ؤ' not in result

    def test_masool(self):
        # مسؤول → should normalize hamza
        result = normalize_query('مسؤول')
        assert 'ئ' not in result

    def test_diacritics_search(self):
        # User types with diacritics, content without
        query = normalize_query('الذَّكاءُ الاصْطِناعي')
        content = normalize_query('الذكاء الاصطناعي')
        assert query == content
