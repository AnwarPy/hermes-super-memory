#!/usr/bin/env python3
"""
Hermes Memory Decision Engine — SQLite Edition (P2)

Mem0-style ADD/UPDATE/DELETE/NOOP resolver.
Reads from SQLite facts table instead of NetworkX graph.json.

P2: For each candidate fact, retrieve top-k similar existing facts,
then a short LLM call decides: ADD (new), UPDATE (refinement),
DELETE+ADD (contradiction), or NOOP (already known).
"""

import json
import os
import re
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime, timezone

# ============================================================
# Configuration
# ============================================================
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.getenv("HERMES_SUMMARIZER_MODEL", "qwen2.5:7b")
DB_PATH = os.path.expanduser("~/.hermes/memory/hermes_memory.db")

VALID_CATEGORIES = (
    "preference", "fact", "decision", "correction",
    "project", "technical", "personal", "service", "general"
)

# ============================================================
# Arabic normalization (import from quality_gates if available)
# ============================================================
_script_dir = str(Path(__file__).parent)
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

try:
    from quality_gates import normalize_arabic_text
except ImportError:
    def normalize_arabic_text(text: str) -> str:
        if not text:
            return text
        text = re.sub(r'[\u064B-\u065F\u0610-\u061A\u0670]', '', text)
        text = re.sub(r'[إأآٱ]', 'ا', text)
        text = re.sub(r'ى', 'ي', text)
        text = re.sub(r'ة', 'ه', text)
        return re.sub(r'\s+', ' ', text).strip()

# ============================================================
# Embedding
# ============================================================
_embedding_model = None

def _get_embedding_model():
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model
    
    plugin_dir = os.path.expanduser("~/.hermes/plugins")
    if plugin_dir not in sys.path:
        sys.path.insert(0, plugin_dir)
    
    try:
        from unified.embedding_model import EmbeddingModel
        _embedding_model = EmbeddingModel(model_name="BAAI/bge-m3", device="cpu", use_fp16=False)
    except ImportError:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer("BAAI/bge-m3", device="cpu")
    
    return _embedding_model


def embed(text):
    """Get BGE-M3 embedding."""
    model = _get_embedding_model()
    if hasattr(model, 'embed_query'):
        return model.embed_query(text)
    return model.encode([text], normalize_embeddings=True)[0].tolist()

# ============================================================
# DB access (SQLite, NOT NetworkX)
# ============================================================
_db = None

def _get_db():
    """Lazy-load MemoryDB singleton."""
    global _db
    if _db is None:
        from db import MemoryDB
        _db = MemoryDB(DB_PATH)
        _db.init()
    return _db

# ============================================================
# Find similar facts (SQLite-based)
# ============================================================

def find_similar_facts(candidate_text: str, top_k: int = 3) -> list:
    """
    Find top-k similar existing facts via cosine similarity.
    Reads from SQLite facts table (live facts with embeddings).
    """
    import numpy as np
    
    db = _get_db()
    
    # Normalize candidate
    normalized_text = normalize_arabic_text(candidate_text)
    cand_emb = embed(normalized_text)
    
    # Use db.search_similar for efficient vector search
    results = db.search_similar(cand_emb, top_k=top_k, threshold=0.5)
    
    # Format to match old API
    return [
        {
            'node_id': r['id'],
            'content': r['key'],
            'similarity': r['similarity'],
        }
        for r in results
    ]


# ============================================================
# LLM Call
# ============================================================

def _call_ollama(prompt, timeout=30):
    """Call Ollama API for decision."""
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "temperature": 0.1,
    }).encode()
    
    req = urllib.request.Request(
        OLLAMA_URL, data=payload,
        headers={"Content-Type": "application/json"}
    )
    
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            return data.get("message", {}).get("content", "")
    except Exception as e:
        print(f"  ⚠ Ollama call failed: {e}")
        return None


# ============================================================
# Decision logic
# ============================================================

def decide(candidate_fact, similar_facts):
    """Decide: ADD / UPDATE / DELETE+ADD / NOOP.
    
    Args:
        candidate_fact: dict with 'key', 'category', etc.
        similar_facts: list of {'node_id': int, 'content': str, 'similarity': float}
    
    Returns: (decision, target_node_id_or_none, reason)
        decision ∈ {'add', 'update', 'contradict', 'noop'}
        target_node_id: SQLite fact id (int) or None
    """
    if not similar_facts:
        return ("add", None, "No similar facts found — new knowledge")
    
    max_sim = similar_facts[0]['similarity']
    
    # If very high similarity (>0.95), likely exact duplicate → NOOP
    if max_sim > 0.95:
        return ("noop", similar_facts[0]['node_id'], 
                f"Near-duplicate of existing fact (sim={max_sim:.3f})")
    
    # For moderate similarity (0.5-0.95), ask LLM
    similar_text = "\n".join(
        f"  [{i+1}] (sim={f['similarity']:.2f}): {f['content']}"
        for i, f in enumerate(similar_facts[:3])
    )
    
    is_ar = bool(re.search(r'[\u0600-\u06FF]', candidate_fact.get('key', '')))
    
    if is_ar:
        prompt = f"""أنت نظام ذاكرة. قرر إذا كانت الحقيقة الجديدة يجب أن تُضاف أم لا.

الحقيقة الجديدة: "{candidate_fact['key']}"
الفئة: {candidate_fact.get('category', 'general')}

حقائق مشابهة موجودة:
{similar_text}

القرارات الممكنة:
- "add": حقيقة جديدة مختلفة، أضفها
- "update": الحقيقة الجديدة نسخة محسّنة/أدق من حقيقة موجودة، استبدلها
- "contradict": الحقيقة الجديدة تناقض حقيقة موجودة، أبطل القديمة وأضف الجديدة
- "noop": الحقيقة موجودة أصلاً بشكل كافٍ، تجاهل الجديدة

أجب بـ JSON فقط: {{"decision": "...", "target_node_id": "..." أو null, "reason": "..."}}"""
    else:
        prompt = f"""You are a memory system. Decide what to do with a new fact.

New fact: "{candidate_fact['key']}"
Category: {candidate_fact.get('category', 'general')}

Similar existing facts:
{similar_text}

Possible decisions:
- "add": New, different fact — add it
- "update": New fact is a refinement/correction of existing — replace it  
- "contradict": New fact contradicts existing — invalidate old, add new
- "noop": Already known sufficiently — skip

Respond with JSON only: {{"decision": "...", "target_node_id": "..." or null, "reason": "..."}}"""

    response = _call_ollama(prompt)
    if response is None:
        return ("add", None, "LLM call failed, defaulting to add")
    
    # Parse decision
    decision = "add"
    target = None
    reason = response[:200]
    
    try:
        # Extract JSON from response
        json_match = re.search(r'\{[^{}]+\}', response, re.DOTALL)
        if json_match:
            d = json.loads(json_match.group())
            decision = d.get("decision", "add").lower()
            target = d.get("target_node_id")
            reason = d.get("reason", response[:200])
    except (json.JSONDecodeError, AttributeError):
        pass
    
    # Normalize decision
    if decision in ("add", "new"):
        decision = "add"
    elif decision in ("update", "refine", "modify"):
        decision = "update"
    elif decision in ("contradict", "contradiction", "delete"):
        decision = "contradict"
    else:
        decision = "noop"
    
    return (decision, target, reason)


# ============================================================
# High-level API: process a candidate fact end-to-end
# ============================================================

def process_fact(candidate_fact: dict) -> dict:
    """
    Process a single candidate fact through the decision engine.
    
    Args:
        candidate_fact: {'key': str, 'category': str, 'importance': int, ...}
    
    Returns:
        {'decision': str, 'action': str, 'fact_id': int|None, 'reason': str}
    """
    similar = find_similar_facts(candidate_fact['key'], top_k=3)
    decision, target_id, reason = decide(candidate_fact, similar)
    
    result = {
        'decision': decision,
        'target_id': target_id,
        'reason': reason,
        'similar_count': len(similar),
    }
    
    # Execute the decision on the database
    if decision == 'add':
        db = _get_db()
        try:
            fact_id = db.upsert_fact(
                key=candidate_fact['key'],
                category=candidate_fact.get('category', 'general'),
                embedding=None,  # No embedding yet — will be added by graph_updater
                session_id=candidate_fact.get('session_id', ''),
                source='decision_engine',
                importance=candidate_fact.get('importance', 1),
            )
            result['fact_id'] = fact_id
            result['action'] = 'INSERTED'
        except Exception as e:
            result['action'] = f'ERROR: {e}'
    
    elif decision == 'noop':
        result['fact_id'] = target_id
        result['action'] = 'SKIPPED (already exists)'
    
    elif decision == 'update':
        result['fact_id'] = target_id
        result['action'] = 'UPDATED existing'
    
    elif decision == 'contradict':
        if target_id:
            db = _get_db()
            db.invalidate_fact(target_id)
        result['fact_id'] = target_id
        result['action'] = 'INVALIDATED old (contradiction)'
    
    return result


# ============================================================
# CLI — for testing
# ============================================================
if __name__ == "__main__":
    import sys
    
    # Test with a sample fact
    test_fact = {
        'key': 'المستخدم يفضل لغة بايثون في مشاريع الذكاء الاصطناعي',
        'category': 'preference',
        'importance': 3,
        'session_id': 'test_001',
    }
    
    print("=" * 60)
    print("Memory Decision Engine — SQLite Edition")
    print("=" * 60)
    
    db = _get_db()
    s = db.stats()
    print(f"  DB: {s['live_facts']} live facts")
    
    print(f"\n  Testing fact: {test_fact['key'][:60]}")
    similar = find_similar_facts(test_fact['key'], top_k=3)
    print(f"  Similar facts found: {len(similar)}")
    for sf in similar:
        print(f"    - sim={sf['similarity']:.3f}: {sf['content'][:60]}")
    
    decision, target, reason = decide(test_fact, similar)
    print(f"\n  Decision: {decision.upper()}")
    print(f"  Target: {target}")
    print(f"  Reason: {reason}")
