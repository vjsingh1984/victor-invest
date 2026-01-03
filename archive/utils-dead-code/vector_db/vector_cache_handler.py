#!/usr/bin/env python3
"""
InvestiGator - Vector Cache Handler
Copyright (c) 2025 Vijaykumar Singh
Licensed under the Apache License 2.0

Vector cache handler to integrate vector database with existing cache system
Handles semantic search for unstructured financial text
"""

import logging
import json
from typing import Dict, Any, Optional, Union, Tuple, List
from pathlib import Path
from datetime import datetime

from utils.cache.cache_base import CacheStorageHandler
from utils.cache.cache_types import CacheType
from .vector_engine import FinancialVectorDB, VectorDocument

logger = logging.getLogger(__name__)


class VectorCacheStorageHandler(CacheStorageHandler):
    """
    Vector cache storage handler for semantic search of unstructured financial text.

    This handler stores narrative content (MD&A, risk factors, business descriptions)
    in a vector database for semantic similarity search, while leaving structured
    financial metrics to traditional cache handlers.
    """

    # Define which data types should use vector storage
    VECTOR_ELIGIBLE_TYPES = {
        CacheType.SEC_RESPONSE,  # For narrative sections of SEC filings
        CacheType.LLM_RESPONSE,  # For AI analysis text
    }

    # Patterns to identify narrative content for vectorization
    NARRATIVE_PATTERNS = [
        "management_discussion",
        "risk_factors",
        "business_description",
        "analysis_summary",
        "investment_thesis",
        "key_insights",
        "key_risks",
    ]

    def __init__(self, cache_type: CacheType, base_path: str = None, priority: int = 5, config=None):
        super().__init__(cache_type, priority)

        self.config = config
        self.base_path = Path(base_path or f"data/vector_cache/{cache_type.value}")
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Initialize vector database
        embedding_model = self._get_embedding_model()
        self.vector_db = FinancialVectorDB(str(self.base_path), embedding_model=embedding_model)

        logger.info(f"VectorCacheStorageHandler initialized for {cache_type.value} at {self.base_path}")

    def _get_embedding_model(self) -> str:
        """Get embedding model from config or use default"""
        if self.config and hasattr(self.config, "vector_db"):
            return getattr(self.config.vector_db, "embedding_model", "all-MiniLM-L6-v2")
        return "all-MiniLM-L6-v2"

    def get(self, key: Union[Tuple, Dict]) -> Optional[Dict[str, Any]]:
        """
        Get data from vector cache.
        For vector storage, this means searching for semantically similar content.
        """
        try:
            # Convert key to search parameters
            symbol, search_context = self._extract_search_params(key)

            if not symbol or not search_context:
                logger.debug(f"Vector cache: insufficient search params from key {key}")
                return None

            # Search for documents
            doc_type = self._get_doc_type_from_cache_type()
            documents = self.vector_db.get_documents_by_symbol(symbol, doc_type)

            if not documents:
                logger.debug(f"Vector cache MISS: No documents found for {symbol}")
                return None

            # Return most recent document or aggregate results
            if len(documents) == 1:
                result = self._document_to_cache_format(documents[0])
                logger.info(f"Vector cache HIT: Found 1 document for {symbol}")
                return result
            else:
                # Multiple documents - return aggregated view
                result = self._aggregate_documents(documents)
                logger.info(f"Vector cache HIT: Found {len(documents)} documents for {symbol}")
                return result

        except Exception as e:
            logger.error(f"Vector cache GET error: {e}")
            return None

    def set(self, key: Union[Tuple, Dict], value: Dict[str, Any]) -> bool:
        """
        Set data in vector cache by extracting narrative content and creating vector documents.
        """
        try:
            # Extract search parameters
            symbol, _ = self._extract_search_params(key)
            if not symbol:
                logger.warning(f"Vector cache: Cannot extract symbol from key {key}")
                return False

            # Check if this data contains narrative content suitable for vectorization
            narrative_content = self._extract_narrative_content(value)

            if not narrative_content:
                logger.debug(f"Vector cache: No narrative content found for {symbol}")
                return False  # Not an error, just not suitable for vector storage

            # Create vector documents for each narrative piece
            documents_created = 0
            for content_type, content_text in narrative_content.items():
                if not content_text or len(content_text.strip()) < 50:
                    continue  # Skip very short content

                # Create vector document
                doc_id = self._generate_doc_id(symbol, content_type, key)
                document = VectorDocument(
                    doc_id=doc_id,
                    content=content_text,
                    doc_type=self._get_doc_type_from_cache_type(),
                    symbol=symbol,
                    fiscal_year=self._extract_fiscal_year(key, value),
                    fiscal_period=self._extract_fiscal_period(key, value),
                    form_type=self._extract_form_type(key, value),
                    topics=[content_type],
                    metadata={
                        "cache_type": self.cache_type.value,
                        "content_type": content_type,
                        "original_key": str(key),
                    },
                )

                # Add to vector database
                if self.vector_db.add_document(document):
                    documents_created += 1
                    logger.debug(f"Vector document created: {doc_id}")

            if documents_created > 0:
                logger.info(f"Vector cache SET: Created {documents_created} documents for {symbol}")
                return True
            else:
                logger.debug(f"Vector cache SET: No documents created for {symbol}")
                return False

        except Exception as e:
            logger.error(f"Vector cache SET error: {e}")
            return False

    def exists(self, key: Union[Tuple, Dict]) -> bool:
        """Check if vector documents exist for the given key"""
        try:
            symbol, _ = self._extract_search_params(key)
            if not symbol:
                return False

            doc_type = self._get_doc_type_from_cache_type()
            documents = self.vector_db.get_documents_by_symbol(symbol, doc_type)
            return len(documents) > 0

        except Exception as e:
            logger.error(f"Vector cache EXISTS error: {e}")
            return False

    def delete(self, key: Union[Tuple, Dict]) -> bool:
        """Delete vector documents for the given key"""
        try:
            symbol, _ = self._extract_search_params(key)
            if not symbol:
                return False

            doc_type = self._get_doc_type_from_cache_type()
            documents = self.vector_db.get_documents_by_symbol(symbol, doc_type)

            deleted_count = 0
            for document in documents:
                if self.vector_db.delete_document(document.doc_id):
                    deleted_count += 1

            logger.info(f"Vector cache DELETE: Removed {deleted_count} documents for {symbol}")
            return deleted_count > 0

        except Exception as e:
            logger.error(f"Vector cache DELETE error: {e}")
            return False

    def delete_by_pattern(self, pattern: str) -> int:
        """Delete vector documents matching a pattern (e.g., symbol prefix)"""
        try:
            # For vector storage, pattern is typically a symbol or symbol prefix
            # Get all documents and filter by pattern
            deleted_count = 0

            # This is a simplified implementation - in production,
            # you might want more sophisticated pattern matching
            doc_types = ["filing", "analysis", "event", "news"]

            for doc_type in doc_types:
                # Get all document keys with prefix (this would need to be implemented in RocksDBVectorStore)
                # For now, we'll use a simple approach
                pass  # Placeholder for pattern-based deletion

            return deleted_count

        except Exception as e:
            logger.error(f"Vector cache DELETE_BY_PATTERN error: {e}")
            return 0

    def clear_all(self) -> bool:
        """Clear all vector documents for this cache type"""
        try:
            # This would require implementing a clear method in FinancialVectorDB
            # For now, return True as placeholder
            logger.warning("Vector cache CLEAR_ALL not fully implemented")
            return True

        except Exception as e:
            logger.error(f"Vector cache CLEAR_ALL error: {e}")
            return False

    def search_similar(self, query: str, symbol: str = None, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Semantic search for similar content.
        This is the main advantage of vector storage over traditional caching.
        """
        try:
            doc_type = self._get_doc_type_from_cache_type()
            results = self.vector_db.search_similar(query=query, doc_type=doc_type, top_k=top_k, symbol=symbol)

            formatted_results = []
            for document, similarity_score in results:
                formatted_results.append(
                    {
                        "document": self._document_to_cache_format(document),
                        "similarity_score": similarity_score,
                        "symbol": document.symbol,
                        "content_type": document.topics[0] if document.topics else "unknown",
                    }
                )

            logger.info(f"Vector search: Found {len(formatted_results)} similar documents for query: {query[:50]}...")
            return formatted_results

        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return []

    def _extract_search_params(self, key: Union[Tuple, Dict]) -> Tuple[Optional[str], Optional[str]]:
        """Extract symbol and search context from cache key"""
        symbol = None
        context = None

        if isinstance(key, dict):
            symbol = key.get("symbol")
            # Build context from other key components
            context_parts = []
            for k, v in key.items():
                if k != "symbol" and v:
                    context_parts.append(f"{k}:{v}")
            context = " ".join(context_parts) if context_parts else None

        elif isinstance(key, tuple) and len(key) > 0:
            symbol = key[0]
            context = " ".join(str(k) for k in key[1:]) if len(key) > 1 else None

        return symbol, context

    def _extract_narrative_content(self, value: Dict[str, Any]) -> Dict[str, str]:
        """Extract narrative content suitable for vectorization"""
        narrative_content = {}

        def extract_from_dict(data: dict, prefix: str = ""):
            for key, val in data.items():
                full_key = f"{prefix}_{key}" if prefix else key

                # Check if this key matches narrative patterns
                if any(pattern in key.lower() for pattern in self.NARRATIVE_PATTERNS):
                    if isinstance(val, str) and len(val.strip()) > 50:
                        narrative_content[full_key] = val.strip()
                elif isinstance(val, dict):
                    extract_from_dict(val, full_key)
                elif isinstance(val, list):
                    # Handle lists of strings (like key_insights, key_risks)
                    if all(isinstance(item, str) for item in val):
                        combined_text = " ".join(val)
                        if len(combined_text.strip()) > 50:
                            narrative_content[full_key] = combined_text.strip()

        # Extract from the value dictionary
        extract_from_dict(value)

        return narrative_content

    def _extract_fiscal_year(self, key: Union[Tuple, Dict], value: Dict[str, Any]) -> Optional[int]:
        """Extract fiscal year from key or value"""
        # Try key first
        if isinstance(key, dict):
            if "fiscal_year" in key:
                return key["fiscal_year"]

        # Try value
        if "fiscal_year" in value:
            return value["fiscal_year"]

        # Fallback to current year
        return datetime.now().year

    def _extract_fiscal_period(self, key: Union[Tuple, Dict], value: Dict[str, Any]) -> Optional[str]:
        """Extract fiscal period from key or value"""
        # Try key first
        if isinstance(key, dict):
            if "fiscal_period" in key:
                return key["fiscal_period"]

        # Try value
        if "fiscal_period" in value:
            return value["fiscal_period"]

        return None

    def _extract_form_type(self, key: Union[Tuple, Dict], value: Dict[str, Any]) -> Optional[str]:
        """Extract form type from key or value"""
        # Try key first
        if isinstance(key, dict):
            if "form_type" in key:
                return key["form_type"]

        # Try value
        if "form_type" in value:
            return value["form_type"]

        return None

    def _generate_doc_id(self, symbol: str, content_type: str, key: Union[Tuple, Dict]) -> str:
        """Generate unique document ID"""
        # Create a hash of the key for uniqueness
        key_hash = hash(str(key)) & 0x7FFFFFFF  # Ensure positive integer
        return f"{symbol}_{content_type}_{key_hash}_{int(datetime.now().timestamp())}"

    def _get_doc_type_from_cache_type(self) -> str:
        """Map cache type to vector document type"""
        mapping = {
            CacheType.SEC_RESPONSE: "filing",
            CacheType.LLM_RESPONSE: "analysis",
        }
        return mapping.get(self.cache_type, "unknown")

    def _document_to_cache_format(self, document: VectorDocument) -> Dict[str, Any]:
        """Convert vector document back to cache format"""
        return {
            "doc_id": document.doc_id,
            "symbol": document.symbol,
            "content": document.content,
            "doc_type": document.doc_type,
            "fiscal_year": document.fiscal_year,
            "fiscal_period": document.fiscal_period,
            "form_type": document.form_type,
            "topics": document.topics,
            "sentiment_score": document.sentiment_score,
            "importance_score": document.importance_score,
            "extraction_date": document.extraction_date.isoformat(),
            "metadata": document.metadata,
        }

    def _aggregate_documents(self, documents: List[VectorDocument]) -> Dict[str, Any]:
        """Aggregate multiple documents into a single cache response"""
        if not documents:
            return {}

        # Sort by extraction date (most recent first)
        sorted_docs = sorted(documents, key=lambda d: d.extraction_date, reverse=True)

        # Use most recent document as base
        latest_doc = sorted_docs[0]

        # Aggregate content from all documents
        aggregated_content = {}
        all_topics = set()

        for doc in sorted_docs:
            if doc.topics:
                all_topics.update(doc.topics)

            # Group content by topic
            for topic in doc.topics:
                if topic not in aggregated_content:
                    aggregated_content[topic] = []
                aggregated_content[topic].append(
                    {"content": doc.content, "extraction_date": doc.extraction_date.isoformat(), "doc_id": doc.doc_id}
                )

        return {
            "symbol": latest_doc.symbol,
            "doc_type": latest_doc.doc_type,
            "fiscal_year": latest_doc.fiscal_year,
            "fiscal_period": latest_doc.fiscal_period,
            "form_type": latest_doc.form_type,
            "total_documents": len(documents),
            "topics": list(all_topics),
            "aggregated_content": aggregated_content,
            "latest_extraction_date": latest_doc.extraction_date.isoformat(),
            "metadata": {"aggregated": True, "document_count": len(documents), "cache_type": self.cache_type.value},
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for this vector cache handler"""
        try:
            base_stats = {
                "handler_type": "VectorCacheStorageHandler",
                "cache_type": self.cache_type.value,
                "priority": self.priority,
                "base_path": str(self.base_path),
            }

            # Get vector database stats
            vector_stats = self.vector_db.get_stats()
            base_stats.update(vector_stats)

            return base_stats

        except Exception as e:
            logger.error(f"Error getting vector cache stats: {e}")
            return {
                "handler_type": "VectorCacheStorageHandler",
                "cache_type": self.cache_type.value,
                "priority": self.priority,
                "error": str(e),
            }
