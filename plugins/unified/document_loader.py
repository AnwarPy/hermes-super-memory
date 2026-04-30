"""Document Loader — تحميل المستندات من المجلدات

يدعم:
- ملفات نصية (.txt, .py, .ts, .js, etc.)
- ملفات PDF
- ملفات Markdown
- فلاتر استبعاد تلقائية (node_modules, .venv, .git, etc.)
- حد أقصى لحجم الملف
"""

from pathlib import Path
from typing import List, Optional, Set
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document

# مجلدات مستبعدة تلقائياً (dependencies + build artifacts)
EXCLUDE_DIRS: Set[str] = {
    '.git',
    'node_modules',
    '.venv',
    'venv',
    'env',
    '__pycache__',
    '.tox',
    '.mypy_cache',
    '.pytest_cache',
    'build',
    'dist',
    '.next',
    'target',
    '.idea',
    '.vscode',
    '.eggs',
}

# حد أقصى لحجم الملف (500KB)
MAX_FILE_SIZE = 500_000


def _should_exclude(file_path: Path) -> bool:
    """تحقق إذا كان الملف في مجلد مستبعد أو حجمه كبير جداً."""
    # تحقق من المجلدات المستبعدة في المسار
    for part in file_path.parts:
        if part in EXCLUDE_DIRS:
            return True
    # تحقق من حجم الملف (فقط لو الملف موجود)
    try:
        if file_path.exists() and file_path.stat().st_size > MAX_FILE_SIZE:
            return True
    except OSError:
        pass  # لو ما نقدر نتحقق من الحجم، نعتمد على فلتر المجلدات فقط
    return False


class DocumentLoader:
    """محمل مستندات متعدد الأنواع مع فلاتر استبعاد"""
    
    def __init__(self, encoding: str = "utf-8"):
        self.encoding = encoding
    
    def load_directory(
        self,
        path: str,
        patterns: Optional[List[str]] = None,
        recursive: bool = True,
    ) -> List[Document]:
        """
        تحميل مستندات من مجلد
        
        Args:
            path: مسار المجلد
            patterns: أنماط الملفات (افتراضي: ["**/*.md", "**/*.py", "**/*.txt"])
            recursive: بحث متداخل
        
        Returns:
            قائمة المستندات المحملة
        """
        if patterns is None:
            patterns = ["**/*.md", "**/*.py", "**/*.txt"]
        
        docs = []
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"المجلد غير موجود: {path}")
        
        for pattern in patterns:
            try:
                if pattern.endswith(".pdf"):
                    # تحميل PDF
                    loader = PyPDFLoader(str(path))
                    pdf_docs = loader.load()
                    # فلترة الملفات المستبعدة
                    for doc in pdf_docs:
                        src = doc.metadata.get("source", "")
                        if src and _should_exclude(Path(src)):
                            continue
                        docs.append(doc)
                elif pattern.endswith(".md"):
                    # تحميل Markdown
                    loader = DirectoryLoader(
                        str(path),
                        glob=pattern,
                        loader_cls=UnstructuredMarkdownLoader,
                        loader_kwargs={"encoding": self.encoding},
                        recursive=recursive,
                    )
                    raw_docs = loader.load()
                    for doc in raw_docs:
                        src = doc.metadata.get("source", "")
                        if src and _should_exclude(Path(src)):
                            continue
                        docs.append(doc)
                else:
                    # تحميل نصوص عادية
                    loader = DirectoryLoader(
                        str(path),
                        glob=pattern,
                        loader_cls=TextLoader,
                        loader_kwargs={"encoding": self.encoding},
                        recursive=recursive,
                    )
                    raw_docs = loader.load()
                    for doc in raw_docs:
                        src = doc.metadata.get("source", "")
                        if src and _should_exclude(Path(src)):
                            continue
                        docs.append(doc)
            except Exception as e:
                print(f"خطأ في تحميل النمط {pattern}: {e}")
                continue
        
        return docs
    
    def load_file(self, file_path: str) -> List[Document]:
        """
        تحميل ملف واحد
        
        Args:
            file_path: مسار الملف
        
        Returns:
            قائمة المستندات (عادة وثيقة واحدة)
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"الملف غير موجود: {file_path}")
        
        if _should_exclude(path):
            raise ValueError(f"الملف في مجلد مستبعد أو كبير جداً: {file_path}")
        
        suffix = path.suffix.lower()
        
        if suffix == ".pdf":
            loader = PyPDFLoader(str(path))
        elif suffix == ".md":
            loader = UnstructuredMarkdownLoader(str(path))
        else:
            loader = TextLoader(str(path), encoding=self.encoding)
        
        return loader.load()
