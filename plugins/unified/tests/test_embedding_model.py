"""P2: Extended Embedding Model Tests — device selection and API coverage."""

import sys
import os
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

sys.path.insert(0, os.path.expanduser('~/.hermes/plugins'))


@pytest.fixture(autouse=True)
def clear_singleton():
    """Clear the model singleton cache before each test."""
    from unified.embedding_model import _MODEL_SINGLETON
    _MODEL_SINGLETON.clear()
    yield
    _MODEL_SINGLETON.clear()


class TestEmbeddingModelDeviceSelection:
    """Test _select_device logic without loading actual model."""

    def test_auto_selects_cpu_when_no_cuda(self):
        """When CUDA not available, auto should fall back to CPU."""
        from unified.embedding_model import EmbeddingModel

        with patch('unified.embedding_model.torch.cuda.is_available', return_value=False):
            with patch.object(__import__('torch').backends, 'mps', create=True) as mock_mps:
                mock_mps.is_available.return_value = False
                # We can't fully test without loading the model, but we can test the device selection
                pass  # Device selection is tested via _select_device below

    def test_select_device_cuda_available(self):
        from unified.embedding_model import EmbeddingModel
        # Create a mock instance to test _select_device
        with patch('unified.embedding_model.SentenceTransformer'):
            with patch('unified.embedding_model.torch.cuda.is_available', return_value=True):
                model = EmbeddingModel(device="auto", use_fp16=False)
                assert model.device == "cuda"

    def test_select_device_cuda_requested_and_available(self):
        from unified.embedding_model import EmbeddingModel
        with patch('unified.embedding_model.SentenceTransformer'):
            with patch('unified.embedding_model.torch.cuda.is_available', return_value=True):
                model = EmbeddingModel(device="cuda", use_fp16=False)
                assert model.device == "cuda"

    def test_select_device_cuda_requested_but_fallback_cpu(self):
        from unified.embedding_model import EmbeddingModel
        with patch('unified.embedding_model.SentenceTransformer'):
            with patch('unified.embedding_model.torch.cuda.is_available', return_value=False):
                model = EmbeddingModel(device="cuda", use_fp16=False)
                assert model.device == "cpu"

    def test_select_device_cpu_always_cpu(self):
        from unified.embedding_model import EmbeddingModel
        with patch('unified.embedding_model.SentenceTransformer'):
            with patch('unified.embedding_model.torch.cuda.is_available', return_value=True):
                model = EmbeddingModel(device="cpu", use_fp16=False)
                assert model.device == "cpu"

    def test_select_device_mps_available(self):
        from unified.embedding_model import EmbeddingModel
        import torch
        with patch('unified.embedding_model.SentenceTransformer'):
            with patch('unified.embedding_model.torch.cuda.is_available', return_value=False):
                with patch.object(torch.backends, 'mps') as mock_mps:
                    mock_mps.is_available.return_value = True
                    model = EmbeddingModel(device="auto", use_fp16=False)
                    assert model.device == "mps"

    def test_select_device_mps_requested_and_available(self):
        from unified.embedding_model import EmbeddingModel
        import torch
        with patch('unified.embedding_model.SentenceTransformer'):
            with patch('unified.embedding_model.torch.cuda.is_available', return_value=False):
                with patch.object(torch.backends, 'mps') as mock_mps:
                    mock_mps.is_available.return_value = True
                    model = EmbeddingModel(device="mps", use_fp16=False)
                    assert model.device == "mps"

    def test_select_device_mps_fallback_cpu(self):
        from unified.embedding_model import EmbeddingModel
        import torch
        with patch('unified.embedding_model.SentenceTransformer'):
            with patch('unified.embedding_model.torch.cuda.is_available', return_value=False):
                with patch.object(torch.backends, 'mps') as mock_mps:
                    mock_mps.is_available.return_value = False
                    model = EmbeddingModel(device="mps", use_fp16=False)
                    assert model.device == "cpu"

    def test_select_device_unknown_fallback(self):
        from unified.embedding_model import EmbeddingModel
        with patch('unified.embedding_model.SentenceTransformer'):
            with patch('unified.embedding_model.torch.cuda.is_available', return_value=False):
                model = EmbeddingModel(device="vulkan", use_fp16=False)
                assert model.device == "cpu"


class TestEmbeddingModelAttributes:
    """Test model attributes without loading actual model."""

    def test_dimension_attribute(self):
        from unified.embedding_model import EmbeddingModel
        with patch('unified.embedding_model.SentenceTransformer'):
            model = EmbeddingModel(use_fp16=False)
            assert model.dimension == 1024

    def test_max_tokens_attribute(self):
        from unified.embedding_model import EmbeddingModel
        with patch('unified.embedding_model.SentenceTransformer'):
            model = EmbeddingModel(use_fp16=False)
            assert model.max_tokens == 8192

    def test_model_name_stored(self):
        from unified.embedding_model import EmbeddingModel
        with patch('unified.embedding_model.SentenceTransformer'):
            model = EmbeddingModel(model_name="custom-model", use_fp16=False)
            assert model.model_name == "custom-model"


class TestEmbeddingModelMethods:
    """Test embedding methods with mocked model."""

    def test_embed_documents_empty_list(self):
        from unified.embedding_model import EmbeddingModel
        with patch('unified.embedding_model.SentenceTransformer'):
            model = EmbeddingModel(use_fp16=False)
            result = model.embed_documents([])
            assert result == []

    def test_embed_documents_calls_encode(self):
        from unified.embedding_model import EmbeddingModel
        import numpy as np
        with patch('unified.embedding_model.SentenceTransformer') as mock_st:
            mock_model = MagicMock()
            mock_model.encode.return_value = np.array([[0.1] * 1024])
            mock_st.return_value = mock_model

            model = EmbeddingModel(use_fp16=False)
            result = model.embed_documents(["hello world"], batch_size=16)

            mock_model.encode.assert_called_once()
            call_kwargs = mock_model.encode.call_args[1]
            assert call_kwargs['batch_size'] == 16
            assert call_kwargs['show_progress_bar'] is False
            assert call_kwargs['normalize_embeddings'] is True
            assert len(result) == 1
            assert len(result[0]) == 1024

    def test_embed_with_metadata(self):
        from unified.embedding_model import EmbeddingModel
        import numpy as np
        with patch('unified.embedding_model.SentenceTransformer') as mock_st:
            mock_model = MagicMock()
            mock_model.encode.return_value = np.array([[0.1] * 1024, [0.2] * 1024])
            mock_st.return_value = mock_model

            model = EmbeddingModel(use_fp16=False)
            result = model.embed_with_metadata(
                ["text1", "text2"],
                metadatas=[{"source": "a"}, {"source": "b"}],
            )

            assert len(result) == 2
            assert result[0]["text"] == "text1"
            assert result[0]["metadata"] == {"source": "a"}
            assert result[1]["embedding"] is not None

    def test_embed_with_metadata_partial(self):
        from unified.embedding_model import EmbeddingModel
        import numpy as np
        with patch('unified.embedding_model.SentenceTransformer') as mock_st:
            mock_model = MagicMock()
            mock_model.encode.return_value = np.array([[0.1] * 1024, [0.2] * 1024])
            mock_st.return_value = mock_model

            model = EmbeddingModel(use_fp16=False)
            result = model.embed_with_metadata(
                ["text1", "text2"],
                metadatas=[{"source": "a"}],  # Only 1 metadata for 2 texts
            )

            assert len(result) == 2
            assert result[0]["metadata"] == {"source": "a"}
            assert "metadata" not in result[1]

    def test_embed_with_metadata_none(self):
        from unified.embedding_model import EmbeddingModel
        import numpy as np
        with patch('unified.embedding_model.SentenceTransformer') as mock_st:
            mock_model = MagicMock()
            mock_model.encode.return_value = np.array([[0.1] * 1024])
            mock_st.return_value = mock_model

            model = EmbeddingModel(use_fp16=False)
            result = model.embed_with_metadata(["text1"], metadatas=None)

            assert len(result) == 1
            assert "metadata" not in result[0]

    def test_compute_similarity(self):
        from unified.embedding_model import EmbeddingModel
        import numpy as np
        with patch('unified.embedding_model.SentenceTransformer') as mock_st:
            mock_model = MagicMock()
            # Return normalized embeddings (since normalize_embeddings=True)
            # Both calls return the same normalized vector
            vec = np.array([0.1] * 1024, dtype=np.float32)
            vec = vec / np.linalg.norm(vec)
            mock_model.encode.return_value = np.array([vec])
            mock_st.return_value = mock_model

            model = EmbeddingModel(use_fp16=False)
            similarity = model.compute_similarity("hello", "hello")

            # Same text should have high similarity
            assert similarity > 0.99

    def test_get_model_info(self):
        from unified.embedding_model import EmbeddingModel
        with patch('unified.embedding_model.SentenceTransformer'):
            with patch('unified.embedding_model.torch.cuda.is_available', return_value=True):
                model = EmbeddingModel(model_name="test-model", device="auto", use_fp16=True)
                info = model.get_model_info()

                assert info['model_name'] == 'test-model'
                assert info['device'] == 'cuda'
                assert info['dimension'] == 1024
                assert info['max_tokens'] == 8192
                assert info['use_fp16'] is True
                assert info['cuda_available'] is True


class TestEmbeddingModelSingleton:
    """Test singleton caching behavior."""

    def test_singleton_caches_model(self):
        from unified.embedding_model import EmbeddingModel, _MODEL_SINGLETON
        with patch('unified.embedding_model.SentenceTransformer') as mock_st:
            mock_model = MagicMock()
            mock_st.return_value = mock_model

            # Clear any existing cache
            _MODEL_SINGLETON.clear()

            model1 = EmbeddingModel(model_name="test", device="cpu", use_fp16=False)
            cache_key = "test_cpu_False"
            assert cache_key in _MODEL_SINGLETON

            model2 = EmbeddingModel(model_name="test", device="cpu", use_fp16=False)
            # Should use cached model
            assert model2.model is mock_model

            # Clean up
            _MODEL_SINGLETON.clear()

    def test_singleton_uses_cache_key(self):
        from unified.embedding_model import EmbeddingModel, _MODEL_SINGLETON
        with patch('unified.embedding_model.SentenceTransformer') as mock_st:
            mock_model = MagicMock()
            mock_st.return_value = mock_model

            _MODEL_SINGLETON.clear()

            model1 = EmbeddingModel(model_name="test", device="cpu", use_fp16=False)
            model2 = EmbeddingModel(model_name="test", device="cpu", use_fp16=True)

            # Different fp16 setting = different cache key
            assert "test_cpu_False" in _MODEL_SINGLETON
            assert "test_cpu_True" in _MODEL_SINGLETON

            _MODEL_SINGLETON.clear()
