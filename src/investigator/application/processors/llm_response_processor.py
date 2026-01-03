#!/usr/bin/env python3
"""
InvestiGator - Common LLM Response Processor
Copyright (c) 2025 Vijaykumar Singh
Licensed under the Apache License 2.0

Common utilities for processing LLM responses across all modules.
Handles thinking tags, JSON extraction, and response normalization.
This ensures consistent handling of both cached and direct LLM responses.
"""

import json
import re
import logging
from typing import Dict, Any, Tuple, Optional, Union

logger = logging.getLogger(__name__)


class LLMResponseProcessor:
    """Common processor for all LLM responses"""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._last_thinking_content = ""

    def process_response(self, response_data: Any, from_cache: bool = False) -> Tuple[str, Dict[str, Any]]:
        """
        Process LLM response uniformly whether from cache or direct generation.

        Args:
            response_data: Response data from cache or LLM
            from_cache: Boolean indicating if response is from cache

        Returns:
            Tuple of (processed_content, metadata)
        """
        try:
            metadata = {
                "from_cache": from_cache,
                "processing_time_ms": 0,
                "has_thinking": False,
                "thinking_content": "",
            }

            if from_cache:
                # Extract from cache structure
                content, cache_metadata = self._extract_from_cache(response_data)
                metadata.update(cache_metadata)

                # Check if content is already processed
                if self._is_already_processed(response_data, content):
                    self.logger.debug("Content is already processed, skipping processing")
                    return content, metadata

            else:
                # Extract from direct LLM response
                content = self._extract_from_llm_response(response_data)
                # Preserve any metadata from the response
                if isinstance(response_data, dict):
                    if "processing_metadata" in response_data:
                        metadata.update(response_data["processing_metadata"])
                    if "metadata" in response_data:
                        metadata.update(response_data["metadata"])

            # Clean content (remove thinking tags, etc.) - only for raw content
            cleaned_content, thinking_content = self._clean_content(content)

            if thinking_content:
                metadata["has_thinking"] = True
                metadata["thinking_content"] = thinking_content
                self._last_thinking_content = thinking_content

            return cleaned_content, metadata

        except Exception as e:
            self.logger.error(f"Error processing LLM response: {e}")
            # Return original content as fallback
            if from_cache:
                return str(response_data.get("response", "")), metadata
            else:
                return str(response_data), metadata

    def _extract_from_cache(self, cache_data: Dict) -> Tuple[str, Dict]:
        """Extract content and metadata from cached response"""
        metadata = cache_data.get("metadata", {})
        response_obj = cache_data.get("response", {})

        # Handle different response object formats
        if isinstance(response_obj, str):
            # Try to parse as JSON
            try:
                response_obj = json.loads(response_obj)
            except:
                # If not JSON, use as is
                return response_obj, metadata

        # Extract content from response object
        if isinstance(response_obj, dict):
            # Check for type field (from prompt_manager)
            if response_obj.get("type") == "text":
                content = response_obj.get("content", "")
            else:
                content = response_obj.get("content", str(response_obj))
        else:
            content = str(response_obj)

        return content, metadata

    def _extract_from_llm_response(self, response_data: Any) -> str:
        """Extract content from direct LLM response"""
        if isinstance(response_data, dict):
            # Check various possible content locations
            if "content" in response_data:
                return response_data.get("content", "")
            elif "response" in response_data:
                return response_data.get("response", "")
            elif "overall_score" in response_data:
                # Already parsed synthesis content
                return json.dumps(response_data, ensure_ascii=False)
            elif "technical_score" in response_data:
                # Already parsed technical content
                return json.dumps(response_data, ensure_ascii=False)
            elif "financial_health_score" in response_data:
                # Already parsed fundamental content
                return json.dumps(response_data, ensure_ascii=False)
            else:
                # Assume the dict itself is the content
                return json.dumps(response_data, ensure_ascii=False)
        else:
            return str(response_data)

    def _clean_content(self, content: str) -> Tuple[str, str]:
        """
        Clean content by removing thinking tags and other problematic elements.

        Returns:
            Tuple of (cleaned_content, thinking_content)
        """
        if not content:
            return content, ""

        # Convert to string if not already
        if not isinstance(content, str):
            content = str(content)

        # Debug logging
        self.logger.debug(f"_clean_content input type: {type(content)}")
        self.logger.debug(f"_clean_content input length: {len(content)}")
        self.logger.debug(f"_clean_content input preview: {content[:200]}...")

        # Extract and remove thinking tags
        thinking_content = ""
        think_pattern = r"<think>(.*?)</think>"
        think_matches = re.findall(think_pattern, content, re.DOTALL)

        if think_matches:
            thinking_content = "\n\n".join(think_matches)
            content = re.sub(think_pattern, "", content, flags=re.DOTALL).strip()
            self.logger.debug(f"Removed {len(think_matches)} thinking sections from response")

        # Fix common JSON escape sequence issues from reasoning models
        # This handles cases where LLM outputs invalid escape sequences
        backslash_present = "\\" in content
        self.logger.debug(f"Checking for backslashes: {backslash_present}")
        if backslash_present:
            try:
                # First, try direct JSON parsing
                json.loads(content)
                # If successful, no fixes needed
                self.logger.debug("JSON parsing successful, no fixes needed")
            except json.JSONDecodeError as e:
                # JSON parsing failed, try to fix the specific issues
                self.logger.debug(f"JSON parse failed at position {e.pos}: {e}")

                # Extract the problematic area to understand the issue
                if e.pos < len(content):
                    error_context = content[max(0, e.pos - 50) : e.pos + 50]
                    self.logger.debug(f"Error context: {repr(error_context)}")

                # Fix the most common issue: unescaped quotes in string values
                # Replace \" with \\" within JSON string values, but be careful not to break valid escapes
                fixed_content = content

                # Pattern: find "field": "...content with \"quotes\"..." and fix the quotes
                # This is a simple approach: replace \" with \\" but only if it's not already \\\"
                fixed_content = re.sub(r'(?<!\\)\\"', r'\\\\"', fixed_content)

                # Try parsing the fixed content
                try:
                    json.loads(fixed_content)
                    content = fixed_content
                    self.logger.debug("Successfully fixed JSON by escaping quotes")
                except json.JSONDecodeError:
                    # Still fails, try more aggressive fixes
                    # Replace literal newlines and tabs in JSON strings
                    fixed_content = re.sub(r'(": "[^"]*?)\\n([^"]*?")', r"\1\\\\n\2", content)
                    fixed_content = re.sub(r'(": "[^"]*?)\\t([^"]*?")', r"\1\\\\t\2", fixed_content)

                    try:
                        json.loads(fixed_content)
                        content = fixed_content
                        self.logger.debug("Successfully fixed JSON by escaping newlines/tabs")
                    except json.JSONDecodeError:
                        # Last resort: leave as is and let validation handle it
                        self.logger.warning("Could not fix JSON escape sequences, proceeding with original")
            except Exception as e:
                # For any other errors, leave content as is
                self.logger.debug(f"Error in JSON fixing: {e}")
                pass

        # Debug logging for output
        self.logger.debug(f"_clean_content output type: {type(content)}")
        self.logger.debug(f"_clean_content output length: {len(content)}")
        self.logger.debug(f"_clean_content output preview: {content[:200]}...")

        return content, thinking_content

    def escape_for_json(self, text: str) -> str:
        """
        Properly escape text content for inclusion in JSON fields.

        Args:
            text: Text to escape

        Returns:
            JSON-safe escaped text
        """
        if not text:
            return text

        # Use json.dumps to properly escape the string, then remove surrounding quotes
        escaped = json.dumps(text)
        # Remove the surrounding quotes that json.dumps adds
        if escaped.startswith('"') and escaped.endswith('"'):
            escaped = escaped[1:-1]

        return escaped

    def extract_json_from_text(self, text: str) -> Optional[Dict]:
        """
        Extract JSON from text that may contain markdown blocks or mixed content.

        Args:
            text: Text potentially containing JSON

        Returns:
            Extracted JSON dict or None
        """
        if not text:
            return None

        # Try direct JSON parsing first
        try:
            return json.loads(text)
        except:
            pass

        # Look for JSON in markdown code block
        json_match = re.search(r"```json\s*(.*?)```", text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1).strip())
            except:
                pass

        # Look for JSON-like content (starts with { and ends with })
        brace_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except:
                pass

        return None

    def _is_already_processed(self, response_data: Any, content: str) -> bool:
        """
        Check if the response content has already been processed.

        Args:
            response_data: Original response data structure
            content: Extracted content string

        Returns:
            True if content is already processed, False otherwise
        """
        try:
            # Check for processed flag in metadata
            if isinstance(response_data, dict):
                # Check response object for processed flag
                response_obj = response_data.get("response", {})
                if isinstance(response_obj, dict) and response_obj.get("processed"):
                    return True

                # Check metadata for processed flag
                metadata = response_data.get("metadata", {})
                if isinstance(metadata, dict) and metadata.get("processed"):
                    return True

            # Check if content looks like clean JSON (already processed)
            if content.strip().startswith("{") and content.strip().endswith("}"):
                try:
                    parsed = json.loads(content)
                    # If it parses as JSON and doesn't contain <think> tags or problematic escapes,
                    # it's likely already processed
                    if "<think>" not in content and '\\"' not in content:
                        return True
                except json.JSONDecodeError:
                    pass

            return False

        except Exception as e:
            self.logger.debug(f"Error checking if content is processed: {e}")
            return False

    @property
    def last_thinking_content(self) -> str:
        """Get the last extracted thinking content"""
        return self._last_thinking_content


# Singleton instance
_processor = None


def get_llm_response_processor() -> LLMResponseProcessor:
    """Get or create the singleton LLM response processor"""
    global _processor
    if _processor is None:
        _processor = LLMResponseProcessor()
    return _processor
