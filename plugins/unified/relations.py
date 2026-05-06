"""P3: Typed Graph Relations — LLM-based relation classifier.

Adds typed, directed edges between facts:
- causes      → A يسبب B
- fixes       → A يحل B  
- supports    → A يدعم B
- contradicts → A يناقض B

Uses Ollama (qwen2.5:3b) to classify relations between fact pairs.
Falls back gracefully when Ollama is unavailable.

Usage:
    from unified.relations import RelationClassifier

    classifier = RelationClassifier(config={
        'ollama_url': 'http://localhost:11434/api/generate',
        'relation_model': 'qwen2.5:3b',
        'confidence_threshold': 0.7,
        'fallback_on_error': True,
    })

    # Classify relation between two facts
    result = classifier.classify_relation(
        fact_a="المستخدم يستخدم SQLite على جهاز محلي",
        fact_b="الأداء سريع لأنه ما يحتاج سيرفر خارجي",
    )
    # Returns: {'relation_type': 'causes', 'confidence': 0.85}

    # Add to database
    if result and result['confidence'] >= 0.7:
        db.add_relation(from_id=a_id, to_id=b_id, 
                        kind=result['relation_type'],
                        weight=result['confidence'])
"""

import json
import logging
import os
from typing import Optional, Dict, List, Any

logger = logging.getLogger(__name__)

# ============================================================
# Relation type definitions
# ============================================================

RELATION_TYPES = {
    'causes': {
        'description_ar': 'السبب يؤدي إلى النتيجة',
        'description_en': 'A causes B',
        'direction': 'asymmetric',  # A→B ≠ B→A
    },
    'fixes': {
        'description_ar': 'الحل يعالج المشكلة',
        'description_en': 'A fixes B',
        'direction': 'asymmetric',
    },
    'supports': {
        'description_ar': 'أ تدعم ب',
        'description_en': 'A supports B',
        'direction': 'asymmetric',
    },
    'contradicts': {
        'description_ar': 'أ تناقض ب',
        'description_en': 'A contradicts B',
        'direction': 'symmetric',  # A↔B
    },
    'related': {
        'description_ar': 'أ لها علاقة بـ ب',
        'description_en': 'A is related to B',
        'direction': 'symmetric',
    },
}

VALID_TYPES = set(RELATION_TYPES.keys())

# ============================================================
# LLM prompt template
# ============================================================

RELATION_CLASSIFY_PROMPT = """Given two facts, determine the relationship between them.

Available relationship types:
- causes: Fact A causes or leads to Fact B
- fixes: Fact A fixes or resolves Fact B (which is a problem/error)
- supports: Fact A supports, enables, or strengthens Fact B
- contradicts: Fact A contradicts or conflicts with Fact B
- related: Fact A is related to Fact B but none of the above apply

Fact A: {fact_a}
Fact B: {fact_b}

Output ONLY a JSON object with:
- relation_type: one of [causes, fixes, supports, contradicts, related, none]
- confidence: a float between 0.0 and 1.0
- reasoning: a brief explanation in Arabic

Example output:
{{"relation_type": "causes", "confidence": 0.85, "reasoning": "الحقيقة الأولى تصف السبب والثانية النتيجة"}}
"""


class RelationClassifier:
    """P3: LLM-based relation classifier with Ollama fallback."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.ollama_url = self.config.get(
            'ollama_url', 'http://localhost:11434/api/generate'
        )
        self.relation_model = self.config.get('relation_model', 'qwen2.5:3b')
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        self.fallback_on_error = self.config.get('fallback_on_error', True)

    def classify_relation(self, fact_a: str, fact_b: str) -> Optional[Dict]:
        """Classify the relationship between two facts.
        
        Returns dict with relation_type, confidence, reasoning or None on failure.
        """
        try:
            return self._classify_with_llm(fact_a, fact_b)
        except Exception as e:
            logger.warning("LLM classification failed: %s", e)
            if self.fallback_on_error:
                return self._classify_simple(fact_a, fact_b)
            return None

    def _classify_with_llm(self, fact_a: str, fact_b: str) -> Optional[Dict]:
        """Use Ollama to classify the relation."""
        import httpx

        prompt = RELATION_CLASSIFY_PROMPT.format(fact_a=fact_a, fact_b=fact_b)

        response = httpx.post(
            self.ollama_url,
            json={
                'model': self.relation_model,
                'prompt': prompt,
                'stream': False,
                'temperature': 0.1,  # Low temperature for consistent classification
            },
            timeout=15.0,
        )
        response.raise_for_status()
        result = response.json()
        text = result.get('response', '').strip()

        # Extract JSON from response (might have markdown formatting)
        text = text.strip().strip('`').strip()
        if text.startswith('json'):
            text = text[4:].strip()

        parsed = json.loads(text)

        # Validate
        relation_type = parsed.get('relation_type', 'none').lower()
        confidence = float(parsed.get('confidence', 0.0))
        reasoning = parsed.get('reasoning', '')

        if relation_type not in VALID_TYPES and relation_type != 'none':
            logger.warning("Invalid relation type: %s — defaulting to 'related'", relation_type)
            relation_type = 'related'

        if relation_type == 'none':
            return None

        return {
            'relation_type': relation_type,
            'confidence': confidence,
            'reasoning': reasoning,
        }

    def _classify_simple(self, fact_a: str, fact_b: str) -> Optional[Dict]:
        """Simple keyword-based fallback when LLM is unavailable.
        
        Uses keyword heuristics to guess the relation type.
        Low confidence (always < 0.5) since it's heuristic.
        """
        a_lower = fact_a.lower()
        b_lower = fact_b.lower()

        # Check for fix/problem keywords
        fix_keywords = ['حل', 'fix', 'repair', 'إصلاح', 'معالجة', 'عالج']
        problem_keywords = ['مشكلة', 'خطأ', 'error', 'problem', 'bug', 'فشل']

        has_fix = any(kw in a_lower for kw in fix_keywords)
        has_problem = any(kw in b_lower for kw in problem_keywords)
        if has_fix and has_problem:
            return {'relation_type': 'fixes', 'confidence': 0.35, 'reasoning': 'heuristic'}

        # Check for contradiction
        contradiction_keywords = ['لكن', 'but', 'however', 'عكس', 'تناقض', 'instead']
        if any(kw in a_lower or kw in b_lower for kw in contradiction_keywords):
            return {'relation_type': 'contradicts', 'confidence': 0.3, 'reasoning': 'heuristic'}

        # Check for causation
        cause_keywords = ['بسبب', 'because', 'لذلك', 'therefore', 'أدى', 'causes', 'يؤدي']
        if any(kw in a_lower or kw in b_lower for kw in cause_keywords):
            return {'relation_type': 'causes', 'confidence': 0.35, 'reasoning': 'heuristic'}

        # Default: related (low confidence)
        # Check if they share significant words
        a_words = set(a_lower.split())
        b_words = set(b_lower.split())
        shared = a_words & b_words
        if len(shared) >= 2:
            return {'relation_type': 'related', 'confidence': 0.25, 'reasoning': 'shared_words'}

        return None

    def classify_batch(self, fact_pairs: List[tuple]) -> List[Optional[Dict]]:
        """Classify relations for multiple fact pairs.
        
        Returns list of results (same order as input pairs).
        """
        results = []
        for fact_a, fact_b in fact_pairs:
            results.append(self.classify_relation(fact_a, fact_b))
        return results

    def add_typed_relations(self, db, new_fact_id: int, existing_fact_ids: List[int]) -> int:
        """Classify and add typed relations between a new fact and existing facts.
        
        Returns number of relations added.
        """
        if db is None or not existing_fact_ids:
            return 0

        # Get the new fact text
        conn = getattr(db, 'conn', None)
        if conn is None:
            return 0

        cursor = conn.execute("SELECT key FROM facts WHERE id = ?", [new_fact_id])
        row = cursor.fetchone()
        if not row:
            return 0

        new_fact_key = row[0]
        added = 0

        for existing_id in existing_fact_ids:
            cursor = conn.execute("SELECT key FROM facts WHERE id = ?", [existing_id])
            row = cursor.fetchone()
            if not row:
                continue

            existing_key = row[0]
            result = self.classify_relation(existing_key, new_fact_key)

            if result and result['confidence'] >= self.confidence_threshold:
                db.add_relation(
                    from_id=existing_id,
                    to_id=new_fact_id,
                    kind=result['relation_type'],
                    weight=result['confidence'],
                )
                added += 1
                logger.debug("Added %s relation (conf=%.2f): %s → %s",
                            result['relation_type'], result['confidence'],
                            existing_id, new_fact_id)

        return added
