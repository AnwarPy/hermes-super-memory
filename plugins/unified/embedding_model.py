"""Embedding Model — نموذج التضمين BGE-M3

يدعم:
- BAAI/bge-m3 (1024 أبعاد، دعم عربي ممتاز)
- تشغيل على GPU (CUDA) أو CPU
- FP16 لتوفير VRAM
- ONNX Runtime للسرعة
"""

from typing import List, Optional, Union
import torch
from sentence_transformers import SentenceTransformer

# Global singleton cache — مشاركة النموذج عالمياً
_MODEL_SINGLETON = {}


class EmbeddingModel:
    """نموذج تضمين متعدد اللغات مع دعم عربي ممتاز"""
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "auto",
        use_fp16: bool = True,
        cache_dir: Optional[str] = None,
    ):
        """
        Args:
            model_name: اسم النموذج من HuggingFace
            device: الجهاز (cuda أو cpu)
            use_fp16: استخدام FP16 لتوفير VRAM
            cache_dir: مجلد التخزين المؤقت
        """
        self.model_name = model_name
        self.device = self._select_device(device)
        self.use_fp16 = use_fp16
        
        # معلومات النموذج (دائماً)
        self.dimension = 1024  # BGE-M3 output dimension
        self.max_tokens = 8192  # BGE-M3 context window
        
        # Singleton check — هل النموذج محمّل مسبقاً؟
        cache_key = f"{model_name}_{device}_{use_fp16}"
        if cache_key in _MODEL_SINGLETON:
            print(f"✓ نموذج محمّل مسبقاً: {model_name} (من الكاش)")
            self.model = _MODEL_SINGLETON[cache_key]
        else:
            print(f"جاري تحميل نموذج التضمين: {model_name}")
            print(f"الجهاز: {self.device}")
            
            # تحميل النموذج
            self.model = SentenceTransformer(
                model_name,
                device=self.device,
                model_kwargs={
                    "torch_dtype": torch.float16 if use_fp16 and self.device == "cuda" else torch.float32,
                },
                cache_folder=cache_dir,
            )
            
            # حفظ في الكاش
            _MODEL_SINGLETON[cache_key] = self.model
            print(f"✓ نموذج محمّل: {model_name}")
            print(f"  - الأبعاد: {self.dimension}")
            print(f"  - الحد الأقصى: {self.max_tokens} tokens")
            print(f"  - FP16: {use_fp16}")
    
    def _select_device(self, preferred_device: str) -> str:
        """اختيار الجهاز المتاح
        
        - auto: اختيار تلقائي حسب الأفضلية cuda > mps > cpu
        - cuda/mps/cpu: استخدام المحدد، مع fallback آمن إلى cpu
        """
        if preferred_device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"

        if preferred_device == "cuda" and torch.cuda.is_available():
            return "cuda"
        elif preferred_device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            if preferred_device not in ("cpu", "auto"):
                print(f"تحذير: {preferred_device} غير متاح، جاري استخدام CPU")
            return "cpu"
    
    def embed_query(self, text: str) -> List[float]:
        """
        تحويل نص واحد إلى متجه
        
        Args:
            text: النص للتضمين
        
        Returns:
            متجه التضمين (1024 أبعاد)
        """
        embeddings = self.model.encode(
            [text],
            prompt="Represent this sentence for searching relevant passages: ",
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings[0].tolist()
    
    def embed_documents(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        تحويل قائمة نصوص إلى متجهات
        
        Args:
            texts: قائمة النصوص
            batch_size: حجم الدفعة
        
        Returns:
            قائمة متجهات التضمين
        """
        if not texts:
            return []
        
        embeddings = self.model.encode(
            texts,
            prompt="Represent this sentence for searching relevant passages: ",
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=True,
        )
        
        return embeddings.tolist()
    
    def embed_with_metadata(
        self,
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
        batch_size: int = 32,
    ) -> List[dict]:
        """
        تحويل نصوص مع حفظ البيانات الوصفية
        
        Args:
            texts: قائمة النصوص
            metadatas: قائمة البيانات الوصفية
            batch_size: حجم الدفعة
        
        Returns:
            قائمة dicts مع embedding و metadata
        """
        embeddings = self.embed_documents(texts, batch_size)
        
        results = []
        for i, (text, emb) in enumerate(zip(texts, embeddings)):
            result = {
                "text": text,
                "embedding": emb,
                "dimension": self.dimension,
            }
            if metadatas and i < len(metadatas):
                result["metadata"] = metadatas[i]
            results.append(result)
        
        return results
    
    def compute_similarity(
        self,
        text1: str,
        text2: str,
    ) -> float:
        """
        حساب التشابه بين نصين
        
        Args:
            text1: النص الأول
            text2: النص الثاني
        
        Returns:
            درجة التشابه (0-1)
        """
        emb1 = self.embed_query(text1)
        emb2 = self.embed_query(text2)
        
        # Cosine similarity
        import numpy as np
        v1 = np.array(emb1)
        v2 = np.array(emb2)
        
        similarity = np.dot(v1, v2) / max(np.linalg.norm(v1) * np.linalg.norm(v2), 1e-10)
        return float(similarity)
    
    def get_model_info(self) -> dict:
        """معلومات النموذج"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "dimension": self.dimension,
            "max_tokens": self.max_tokens,
            "use_fp16": self.use_fp16,
            "cuda_available": torch.cuda.is_available(),
        }
