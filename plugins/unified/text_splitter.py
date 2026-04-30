"""Text Splitter — تقسيم النصوص مع وعي بالعربية.

يُستخدم لتقسيم المستندات الكبيرة إلى chunks قابلة للفهرسة.
يدعم بنية الجملة العربية (۔ ؟ ، ؛) والإنجليزية.
"""

from typing import List, Optional
import re


class TextSplitter:
    """تقسيم النصوص مع احترام بنية الجملة العربية والإنجليزية."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 96):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # فواصل تحترم بنية الجملة العربية والإنجليزية
        self.separators = [
            "\n\n", "\n",
            "۔", "؟", "!",        # نقاط نهاية عربية/أردية
            ". ", "? ", "! ",
            "،", ", ", "؛", "; ",
            ":", " ",
            "",
        ]

    def split_text(self, text: str) -> List[str]:
        """تقسيم نص واحد إلى chunks."""
        if not text or not text.strip():
            return []

        if len(text) <= self.chunk_size:
            return [text.strip()]

        chunks = []
        current_chunk = ""

        # تقسيم على الفواصل بالترتيب
        segments = self._split_by_separators(text, self.separators)

        for segment in segments:
            if len(current_chunk) + len(segment) <= self.chunk_size:
                current_chunk += segment
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = segment

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # إضافة overlap
        if self.chunk_overlap > 0 and len(chunks) > 1:
            chunks = self._add_overlap(chunks)

        return chunks

    def _split_by_separators(self, text: str, separators: List[str]) -> List[str]:
        """تقسيم النص باستخدام الفواصل بالترتيب."""
        if not separators:
            return [text]

        sep = separators[0]
        remaining_seps = separators[1:]

        if sep == "":
            # تقسيم حرفي كملاذ أخير
            return [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        parts = text.split(sep)
        result = []

        for i, part in enumerate(parts):
            if not part:
                continue

            if len(part) <= self.chunk_size:
                if i < len(parts) - 1:
                    result.append(part + sep)
                else:
                    result.append(part)
            else:
                # الجزء أطول من chunk_size — قسّم أكثر
                sub_parts = self._split_by_separators(part, remaining_seps)
                result.extend(sub_parts)

        return result

    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """إضافة overlap بين الchunks."""
        if len(chunks) <= 1:
            return chunks

        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            overlap_text = prev[-self.chunk_overlap:]
            # أضف overlap في بداية chunk التالي
            overlapped.append(overlap_text + " " + chunks[i])

        return overlapped

    def split(self, docs, file_type=None) -> list:
        """واجهة متوافقة مع graph_engine.py — يستقبل قائمة Documents.

        الـ ``file_type`` غير مستخدم حالياً ومُحفوظ للتوافق مع
        ``graph_engine.index_directory()`` الذي يمرّره (السطر 130).
        """
        results = []
        for doc in docs:
            if hasattr(doc, "page_content"):
                text = doc.page_content
                metadata = getattr(doc, "metadata", {})
            elif isinstance(doc, dict):
                text = doc.get("page_content", doc.get("content", ""))
                metadata = doc.get("metadata", {})
            else:
                text = str(doc)
                metadata = {}

            chunks = self.split_text(text)
            for chunk in chunks:
                if hasattr(doc, "page_content"):
                    from copy import copy
                    new_doc = copy(doc)
                    new_doc.page_content = chunk
                    results.append(new_doc)
                else:
                    results.append({"page_content": chunk, "metadata": metadata})
        return results


def split_documents(docs: list, chunk_size: int = 512, chunk_overlap: int = 96) -> list:
    """تقسيم قائمة من المستندات (langchain Document objects أو dicts)."""
    splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    results = []

    for doc in docs:
        if hasattr(doc, 'page_content'):
            # langchain Document
            text = doc.page_content
            metadata = doc.metadata if hasattr(doc, 'metadata') else {}
        elif isinstance(doc, dict):
            text = doc.get('content', doc.get('page_content', ''))
            metadata = doc.get('metadata', {})
        else:
            text = str(doc)
            metadata = {}

        chunks = splitter.split_text(text)
        for chunk in chunks:
            if hasattr(doc, 'page_content'):
                # Return same type
                from copy import copy
                new_doc = copy(doc)
                new_doc.page_content = chunk
                results.append(new_doc)
            else:
                results.append({'content': chunk, 'metadata': metadata})

    return results
