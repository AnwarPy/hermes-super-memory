#!/usr/bin/env python3
"""
Hermes Memory Decision Engine — Mem0-style ADD/UPDATE/DELETE/NOOP resolver.

P2: For each candidate fact, retrieve top-k similar existing facts,
then a short LLM call decides: ADD (new), UPDATE (refinement),
DELETE+ADD (contradiction), or NOOP (already known).

This converts the append-only fact log into a self-cleaning knowledge base.
"""

import json
import os
import re
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone

# ============================================================
# Configuration
# ============================================================
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.getenv("HERMES_SUMMARIZER_MODEL", "qwen2.5:7b")

VALID_CATEGORIES = (
    "preference", "fact", "decision", "correction",
    "project", "technical", "personal", "service", "general"
)

# ============================================================
# Arabic normalization (import from quality_gates if available)
# ============================================================
_qg_dir = os.path.dirname(os.path.abspath(__file__))
if _qg_dir not in sys.path:
    sys.path.insert(0, _qg_dir)

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
        # Fallback: use sentence-transformers directly
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer("BAAI/bge-m3", device="cpu")
    
    return _embedding_model


def embed(text):
    model = _get_embedding_model()
    if hasattr(model, 'embed_query'):
        return model.embed_query(text)
    return model.encode([text], normalize_embeddings=True)[0].tolist()

# ============================================================
# Graph loading
# ============================================================
GRAPHS_DIR = os.path.expanduser("~/.hermes/graphs/hermes-memory")

def load_graph():
    """Load the existing graph as (node_ids, embeddings, contents)."""
    graph_path = os.path.join(GRAPHS_DIR, "graph.json")
    if not os.path.exists(graph_path):
        return [], [], []
    
    import networkx as nx
    G = nx.readwrite.json_graph.node_link_graph(json.load(open(graph_path)))
    
    node_ids = []
    embeddings = []
    contents = []
    
    for nid in G.nodes():
        nd = G.nodes[nid]
        if nd.get("type") != "fact":
            continue
        emb = nd.get("embedding")
        content = nd.get("content", "")
        if emb and len(emb) > 0 and content:
            node_ids.append(nid)
            embeddings.append(emb)
            contents.append(content)
    
    return node_ids, embeddings, contents

# ============================================================
# Decision logic
# ============================================================

def find_similar_facts(candidate_text, node_ids, embeddings, contents, top_k=3):
    """Find top-k similar existing facts via cosine similarity."""
    if not embeddings:
        return []
    
    import numpy as np
    
    cand_emb = embed(normalize_arabic_text(candidate_text))
    cand_vec = np.asarray(cand_emb, dtype=np.float32)
    cand_vec = cand_vec / max(np.linalg.norm(cand_vec), 1e-10)
    
    emb_matrix = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    emb_matrix = emb_matrix / np.maximum(norms, 1e-10)
    
    sims = emb_matrix @ cand_vec
    top_indices = np.argsort(-sims)[:top_k]
    
    results = []
    for idx in top_indices:
        if sims[idx] < 0.5:
            break
        results.append({
            'node_id': node_ids[idx],
            'content': contents[idx],
            'similarity': float(sims[idx]),
        })
    
    return results


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


def decide(candidate_fact, similar_facts):
    """Decide: ADD / UPDATE / DELETE+ADD / NOOP.
    
    Returns: (decision, target_node_id_or_none, reason)
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
