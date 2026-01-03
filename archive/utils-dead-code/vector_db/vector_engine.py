#!/usr/bin/env python3
"""
InvestiGator - Vector Database Engine
Copyright (c) 2025 Vijaykumar Singh
Licensed under the Apache License 2.0

Core vector database engine with RocksDB storage and FAISS indexing
Custom implementation for financial data semantic search
"""

import json
import logging
import pickle
import hashlib
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, date
import threading
from abc import ABC, abstractmethod

# Vector and embedding libraries
try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available - vector operations will be disabled")

try:
    import rocksdb

    ROCKSDB_AVAILABLE = True
except ImportError:
    ROCKSDB_AVAILABLE = False
    logging.warning("RocksDB not available - using fallback storage")

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("SentenceTransformers not available - embedding generation disabled")

logger = logging.getLogger(__name__)


@dataclass
class VectorDocument:
    """Represents a document in the vector database"""

    # Document identification
    doc_id: str
    content: str
    doc_type: str  # 'filing', 'analysis', 'event', 'news'

    # Financial context
    symbol: str
    fiscal_year: Optional[int] = None
    fiscal_period: Optional[str] = None
    form_type: Optional[str] = None

    # Vector data
    embedding: Optional[np.ndarray] = None
    embedding_model: Optional[str] = None

    # Content metadata
    content_hash: Optional[str] = None
    content_length: int = 0
    extraction_date: datetime = field(default_factory=datetime.utcnow)

    # Financial metadata
    topics: List[str] = field(default_factory=list)
    sentiment_score: Optional[float] = None
    importance_score: Optional[float] = None

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Calculate derived fields"""
        if not self.content_hash:
            self.content_hash = hashlib.sha256(self.content.encode()).hexdigest()[:16]
        if not self.content_length:
            self.content_length = len(self.content)
        if not self.doc_id:
            self.doc_id = f"{self.symbol}_{self.doc_type}_{self.content_hash}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "doc_id": self.doc_id,
            "content": self.content,
            "doc_type": self.doc_type,
            "symbol": self.symbol,
            "fiscal_year": self.fiscal_year,
            "fiscal_period": self.fiscal_period,
            "form_type": self.form_type,
            "embedding_model": self.embedding_model,
            "content_hash": self.content_hash,
            "content_length": self.content_length,
            "extraction_date": self.extraction_date.isoformat(),
            "topics": self.topics,
            "sentiment_score": self.sentiment_score,
            "importance_score": self.importance_score,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VectorDocument":
        """Create from dictionary"""
        # Handle datetime conversion
        if "extraction_date" in data and isinstance(data["extraction_date"], str):
            data["extraction_date"] = datetime.fromisoformat(data["extraction_date"])

        return cls(**data)


class EmbeddingGenerator:
    """Generates embeddings for financial text using various models"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.embedding_dim = None
        self._model_lock = threading.Lock()

        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self._initialize_model()
        else:
            logger.warning("SentenceTransformers not available - using dummy embeddings")

    def _initialize_model(self):
        """Initialize the embedding model"""
        try:
            with self._model_lock:
                if self.model is None:
                    logger.info(f"Loading embedding model: {self.model_name}")
                    self.model = SentenceTransformer(self.model_name)
                    # Test embedding to get dimensions
                    test_embedding = self.model.encode(["test"])
                    self.embedding_dim = test_embedding.shape[1]
                    logger.info(f"Embedding model loaded. Dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model {self.model_name}: {e}")
            self.model = None

    def generate_embedding(self, text: str) -> Optional[np.ndarray]:
        """Generate embedding for text"""
        if not text or not text.strip():
            return None

        if not SENTENCE_TRANSFORMERS_AVAILABLE or self.model is None:
            # Return dummy embedding for testing
            return np.random.rand(384).astype(np.float32)

        try:
            # Clean and truncate text for embedding
            cleaned_text = self._clean_text_for_embedding(text)
            embedding = self.model.encode([cleaned_text])
            return embedding[0].astype(np.float32)
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None

    def _clean_text_for_embedding(self, text: str, max_length: int = 5000) -> str:
        """Clean and prepare text for embedding generation"""
        # Remove excessive whitespace and special characters
        cleaned = " ".join(text.split())

        # Truncate if too long
        if len(cleaned) > max_length:
            cleaned = cleaned[:max_length]

        return cleaned

    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension"""
        if self.embedding_dim:
            return self.embedding_dim
        elif SENTENCE_TRANSFORMERS_AVAILABLE and self.model:
            return self.model.get_sentence_embedding_dimension()
        else:
            return 384  # Default dimension for all-MiniLM-L6-v2


class RocksDBVectorStore:
    """RocksDB-based storage backend for vector database"""

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db = None
        self._db_lock = threading.RLock()

        if ROCKSDB_AVAILABLE:
            self._initialize_db()
        else:
            logger.warning("RocksDB not available - using in-memory fallback")
            self._memory_store = {}

    def _initialize_db(self):
        """Initialize RocksDB database"""
        try:
            self.db_path.mkdir(parents=True, exist_ok=True)

            opts = rocksdb.Options()
            opts.create_if_missing = True
            opts.max_open_files = 300000
            opts.write_buffer_size = 67108864
            opts.max_write_buffer_number = 3
            opts.target_file_size_base = 67108864
            opts.compression = rocksdb.CompressionType.lz4_compression

            self.db = rocksdb.DB(str(self.db_path), opts)
            logger.info(f"RocksDB initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize RocksDB: {e}")
            self.db = None

    def put(self, key: str, value: bytes) -> bool:
        """Store key-value pair"""
        if not ROCKSDB_AVAILABLE or self.db is None:
            self._memory_store[key] = value
            return True

        try:
            with self._db_lock:
                self.db.put(key.encode(), value)
                return True
        except Exception as e:
            logger.error(f"Failed to put {key}: {e}")
            return False

    def get(self, key: str) -> Optional[bytes]:
        """Retrieve value by key"""
        if not ROCKSDB_AVAILABLE or self.db is None:
            return self._memory_store.get(key)

        try:
            with self._db_lock:
                result = self.db.get(key.encode())
                return result
        except Exception as e:
            logger.error(f"Failed to get {key}: {e}")
            return None

    def delete(self, key: str) -> bool:
        """Delete key-value pair"""
        if not ROCKSDB_AVAILABLE or self.db is None:
            return self._memory_store.pop(key, None) is not None

        try:
            with self._db_lock:
                self.db.delete(key.encode())
                return True
        except Exception as e:
            logger.error(f"Failed to delete {key}: {e}")
            return False

    def get_keys_with_prefix(self, prefix: str) -> List[str]:
        """Get all keys with given prefix"""
        if not ROCKSDB_AVAILABLE or self.db is None:
            return [k for k in self._memory_store.keys() if k.startswith(prefix)]

        keys = []
        try:
            with self._db_lock:
                it = self.db.iterkeys()
                it.seek(prefix.encode())

                for key in it:
                    key_str = key.decode()
                    if key_str.startswith(prefix):
                        keys.append(key_str)
                    else:
                        break
        except Exception as e:
            logger.error(f"Failed to get keys with prefix {prefix}: {e}")

        return keys

    def close(self):
        """Close the database"""
        if self.db:
            with self._db_lock:
                del self.db
                self.db = None


class FinancialVectorDB:
    """Main vector database class for financial data"""

    def __init__(self, db_path: str, embedding_model: str = "all-MiniLM-L6-v2"):
        self.db_path = Path(db_path)
        self.embedding_generator = EmbeddingGenerator(embedding_model)
        self.storage = RocksDBVectorStore(str(self.db_path / "rocksdb"))

        # FAISS indexes by document type
        self.indexes: Dict[str, faiss.Index] = {}
        self.doc_id_to_index: Dict[str, Dict[str, int]] = {}  # doc_type -> {doc_id: faiss_index}
        self.index_to_doc_id: Dict[str, Dict[int, str]] = {}  # doc_type -> {faiss_index: doc_id}

        self._index_lock = threading.RLock()
        self._initialize_indexes()

    def _initialize_indexes(self):
        """Initialize FAISS indexes for different document types"""
        if not FAISS_AVAILABLE:
            logger.warning("FAISS not available - similarity search disabled")
            return

        embedding_dim = self.embedding_generator.get_embedding_dimension()

        # Create indexes for different document types
        doc_types = ["filing", "analysis", "event", "news"]

        for doc_type in doc_types:
            try:
                # Use IndexFlatIP (Inner Product) for cosine similarity
                index = faiss.IndexFlatIP(embedding_dim)
                self.indexes[doc_type] = index
                self.doc_id_to_index[doc_type] = {}
                self.index_to_doc_id[doc_type] = {}
                logger.info(f"FAISS index created for {doc_type} (dim: {embedding_dim})")
            except Exception as e:
                logger.error(f"Failed to create FAISS index for {doc_type}: {e}")

    def add_document(self, document: VectorDocument) -> bool:
        """Add a document to the vector database"""
        try:
            # Generate embedding if not provided
            if document.embedding is None:
                document.embedding = self.embedding_generator.generate_embedding(document.content)
                document.embedding_model = self.embedding_generator.model_name

            if document.embedding is None:
                logger.warning(f"Failed to generate embedding for document {document.doc_id}")
                return False

            # Store document metadata in RocksDB
            doc_key = f"doc:{document.doc_id}"
            doc_data = document.to_dict()
            # Remove embedding from stored data (stored separately in FAISS)
            doc_data.pop("embedding", None)
            doc_bytes = pickle.dumps(doc_data)

            if not self.storage.put(doc_key, doc_bytes):
                logger.error(f"Failed to store document {document.doc_id}")
                return False

            # Add to FAISS index
            if FAISS_AVAILABLE and document.doc_type in self.indexes:
                with self._index_lock:
                    index = self.indexes[document.doc_type]

                    # Normalize embedding for cosine similarity
                    embedding = document.embedding.reshape(1, -1)
                    faiss.normalize_L2(embedding)

                    # Add to index
                    faiss_idx = index.ntotal
                    index.add(embedding)

                    # Update mappings
                    self.doc_id_to_index[document.doc_type][document.doc_id] = faiss_idx
                    self.index_to_doc_id[document.doc_type][faiss_idx] = document.doc_id

            logger.info(f"Document {document.doc_id} added successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to add document {document.doc_id}: {e}")
            return False

    def search_similar(
        self, query: str, doc_type: str = None, top_k: int = 10, symbol: str = None
    ) -> List[Tuple[VectorDocument, float]]:
        """Search for similar documents"""
        if not FAISS_AVAILABLE:
            logger.warning("FAISS not available - cannot perform similarity search")
            return []

        try:
            # Generate query embedding
            query_embedding = self.embedding_generator.generate_embedding(query)
            if query_embedding is None:
                return []

            # Normalize query embedding
            query_embedding = query_embedding.reshape(1, -1)
            faiss.normalize_L2(query_embedding)

            results = []
            doc_types_to_search = [doc_type] if doc_type else list(self.indexes.keys())

            for dt in doc_types_to_search:
                if dt not in self.indexes:
                    continue

                with self._index_lock:
                    index = self.indexes[dt]
                    if index.ntotal == 0:
                        continue

                    # Search in FAISS index
                    scores, indices = index.search(query_embedding, min(top_k, index.ntotal))

                    for score, idx in zip(scores[0], indices[0]):
                        if idx == -1:  # Invalid index
                            continue

                        # Get document ID
                        doc_id = self.index_to_doc_id[dt].get(idx)
                        if not doc_id:
                            continue

                        # Load document
                        document = self.get_document(doc_id)
                        if not document:
                            continue

                        # Filter by symbol if specified
                        if symbol and document.symbol != symbol:
                            continue

                        results.append((document, float(score)))

            # Sort by similarity score (descending)
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]

        except Exception as e:
            logger.error(f"Failed to search similar documents: {e}")
            return []

    def get_document(self, doc_id: str) -> Optional[VectorDocument]:
        """Retrieve a document by ID"""
        try:
            doc_key = f"doc:{doc_id}"
            doc_bytes = self.storage.get(doc_key)

            if not doc_bytes:
                return None

            doc_data = pickle.loads(doc_bytes)
            document = VectorDocument.from_dict(doc_data)
            return document

        except Exception as e:
            logger.error(f"Failed to get document {doc_id}: {e}")
            return None

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document from the vector database"""
        try:
            # Get document to find its type
            document = self.get_document(doc_id)
            if not document:
                return False

            # Remove from FAISS index
            if FAISS_AVAILABLE and document.doc_type in self.indexes:
                with self._index_lock:
                    faiss_idx = self.doc_id_to_index[document.doc_type].get(doc_id)
                    if faiss_idx is not None:
                        # FAISS doesn't support efficient deletion, so we mark as deleted
                        # In production, consider rebuilding indexes periodically
                        self.doc_id_to_index[document.doc_type].pop(doc_id, None)
                        self.index_to_doc_id[document.doc_type].pop(faiss_idx, None)

            # Remove from RocksDB
            doc_key = f"doc:{doc_id}"
            return self.storage.delete(doc_key)

        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {e}")
            return False

    def get_documents_by_symbol(self, symbol: str, doc_type: str = None) -> List[VectorDocument]:
        """Get all documents for a specific symbol"""
        try:
            # Get all document keys
            prefix = "doc:"
            doc_keys = self.storage.get_keys_with_prefix(prefix)

            documents = []
            for doc_key in doc_keys:
                doc_id = doc_key[len(prefix) :]
                document = self.get_document(doc_id)

                if document and document.symbol == symbol:
                    if doc_type is None or document.doc_type == doc_type:
                        documents.append(document)

            return documents

        except Exception as e:
            logger.error(f"Failed to get documents for symbol {symbol}: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        stats = {
            "embedding_model": self.embedding_generator.model_name,
            "embedding_dimension": self.embedding_generator.get_embedding_dimension(),
            "faiss_available": FAISS_AVAILABLE,
            "rocksdb_available": ROCKSDB_AVAILABLE,
            "doc_types": {},
        }

        if FAISS_AVAILABLE:
            with self._index_lock:
                for doc_type, index in self.indexes.items():
                    stats["doc_types"][doc_type] = {
                        "total_documents": index.ntotal,
                        "index_size_mb": index.ntotal
                        * self.embedding_generator.get_embedding_dimension()
                        * 4
                        / (1024 * 1024),
                    }

        return stats

    def close(self):
        """Close the database"""
        if self.storage:
            self.storage.close()
