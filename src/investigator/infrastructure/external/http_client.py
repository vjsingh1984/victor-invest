# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared HTTP client utilities with proper SSL handling.

Provides:
- SSL-aware aiohttp sessions using certifi certificates
- Retry logic with backoff
- Common headers and timeout handling
"""

import logging
import ssl
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)


def get_ssl_context() -> ssl.SSLContext:
    """Create SSL context with certifi certificates.

    This handles the common SSL certificate verification issues
    on macOS and other systems.
    """
    try:
        import certifi

        ssl_context = ssl.create_default_context(cafile=certifi.where())
        return ssl_context
    except ImportError:
        logger.debug("certifi not available, using default SSL context")
        return ssl.create_default_context()


def get_connector() -> aiohttp.TCPConnector:
    """Get aiohttp connector with proper SSL handling."""
    ssl_context = get_ssl_context()
    return aiohttp.TCPConnector(ssl=ssl_context)


async def create_session() -> aiohttp.ClientSession:
    """Create aiohttp session with proper SSL and headers."""
    connector = get_connector()
    headers = {
        "User-Agent": "Victor-Invest/1.0 (Economic Data Collector)",
        "Accept": "application/json, text/html, application/xhtml+xml, */*",
    }
    return aiohttp.ClientSession(connector=connector, headers=headers)


_shared_session: Optional[aiohttp.ClientSession] = None


async def get_shared_session() -> aiohttp.ClientSession:
    """Get shared aiohttp session (singleton)."""
    global _shared_session
    if _shared_session is None or _shared_session.closed:
        _shared_session = await create_session()
    return _shared_session


async def close_shared_session():
    """Close the shared session."""
    global _shared_session
    if _shared_session and not _shared_session.closed:
        await _shared_session.close()
        _shared_session = None
