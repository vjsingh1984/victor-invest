# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""FastAPI application for Victor Investment API.

This module provides REST API endpoints for investment analysis using
the Victor framework with StateGraph workflows.

Usage:
    uvicorn victor_invest.api.app:app --reload --port 8000

ARCHITECTURE: Victor-Core Integration
=====================================

This API replaces the old InvestiGator REST API with Victor-powered endpoints.
Key differences:

OLD (api/main.py):
- Custom AgentManager with event bus, Redis queues
- Manual agent orchestration
- Workflow IDs tracking state in Redis

NEW (victor_invest/api/app.py):
- StateGraph workflows from Victor framework
- Direct tool invocation (context stuffing pattern)
- In-memory job storage (Redis optional for production)

Endpoints migrated:
- POST /analyze/{symbol} - Single symbol analysis
- POST /batch - Batch analysis
- GET /batch/{job_id} - Batch status
- GET /health - Health check
- GET /cache/stats - Cache statistics
- POST /cache/warm - Warm cache
- DELETE /cache/symbol/{symbol} - Clear symbol cache
- GET /models - List available LLM models
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Victor framework imports
try:
    from victor.framework import Agent
except ImportError:
    Agent = None

from victor_invest.workflows import AnalysisMode, AnalysisWorkflowState, build_graph_for_mode

logger = logging.getLogger(__name__)


# ========================================================================================
# Application Lifecycle
# ========================================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - startup and shutdown."""
    # Startup
    logger.info("Starting Victor Investment API...")

    # Initialize components
    app.state.analysis_jobs = {}
    app.state.cache_manager = None

    # Try to initialize cache manager
    try:
        from investigator.infrastructure.cache import CacheManager

        app.state.cache_manager = CacheManager()
        logger.info("Cache manager initialized")
    except Exception as e:
        logger.warning(f"Cache manager not available: {e}")

    # Try to initialize database
    try:
        from investigator.infrastructure.database import get_database_engine

        app.state.db_engine = get_database_engine()
        logger.info("Database engine initialized")
    except Exception as e:
        logger.warning(f"Database not available: {e}")
        app.state.db_engine = None

    logger.info("Victor Investment API started successfully")

    yield

    # Shutdown
    logger.info("Shutting down Victor Investment API...")

    # Cleanup
    if app.state.cache_manager:
        try:
            # Close cache if it has a close method
            if hasattr(app.state.cache_manager, "close"):
                await app.state.cache_manager.close()
        except Exception as e:
            logger.warning(f"Error closing cache: {e}")

    logger.info("Victor Investment API shutdown complete")


# ========================================================================================
# FastAPI Application
# ========================================================================================

app = FastAPI(
    title="Victor Investment API",
    description="Institutional-grade investment research API powered by Victor AI framework",
    version="0.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GZip compression
app.add_middleware(GZipMiddleware, minimum_size=1000)


# ========================================================================================
# Pydantic Models
# ========================================================================================


class AnalysisRequest(BaseModel):
    """Request model for analysis endpoint."""

    symbol: str = Field(..., description="Stock ticker symbol (e.g., AAPL)")
    mode: str = Field(
        default="standard",
        description="Analysis mode: quick, standard, comprehensive",
    )
    provider: str = Field(default="ollama", description="LLM provider")
    model: Optional[str] = Field(default=None, description="Model name")


class AnalysisResponse(BaseModel):
    """Response model for analysis endpoint."""

    symbol: str
    mode: str
    status: str
    fundamental_analysis: Optional[Dict[str, Any]] = None
    technical_analysis: Optional[Dict[str, Any]] = None
    market_context: Optional[Dict[str, Any]] = None
    synthesis: Optional[Dict[str, Any]] = None
    recommendation: Optional[Dict[str, Any]] = None
    errors: List[str] = []
    timestamp: str


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    version: str
    victor_installed: bool
    providers: List[str]
    services: Dict[str, str]
    timestamp: str


class BatchAnalysisRequest(BaseModel):
    """Request model for batch analysis."""

    symbols: List[str] = Field(..., description="List of stock ticker symbols")
    mode: str = Field(default="standard", description="Analysis mode")


class BatchAnalysisResponse(BaseModel):
    """Response model for batch analysis."""

    submitted: int
    job_id: str
    status: str


class CacheWarmRequest(BaseModel):
    """Request model for cache warming."""

    symbols: List[str] = Field(..., description="Symbols to warm cache for")


class ModelInfo(BaseModel):
    """Model information."""

    name: str
    size: Optional[str] = None
    modified: Optional[str] = None


# ========================================================================================
# API Endpoints
# ========================================================================================


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Victor Investment API",
        "version": "0.2.0",
        "docs": "/docs",
        "status": "operational",
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Comprehensive health check endpoint."""
    victor_installed = Agent is not None

    # Available providers
    providers = ["ollama"]
    if victor_installed:
        providers = ["ollama", "anthropic", "openai", "groq", "deepseek"]

    # Check services
    services = {}

    # Check database
    try:
        from investigator.infrastructure.database import get_database_engine

        engine = get_database_engine()
        services["database"] = "healthy" if engine else "unavailable"
    except Exception:
        services["database"] = "unavailable"

    # Check cache
    try:
        from investigator.infrastructure.cache import CacheManager

        services["cache"] = "healthy"
    except Exception:
        services["cache"] = "unavailable"

    # Check Ollama
    try:
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:11434/api/tags", timeout=2) as resp:
                services["ollama"] = "healthy" if resp.status == 200 else "degraded"
    except Exception:
        services["ollama"] = "unavailable"

    # Determine overall status
    overall_status = "healthy"
    if services.get("ollama") == "unavailable":
        overall_status = "degraded"
    if not victor_installed:
        overall_status = "degraded"

    return HealthResponse(
        status=overall_status,
        version="0.2.0",
        victor_installed=victor_installed,
        providers=providers,
        services=services,
        timestamp=datetime.utcnow().isoformat(),
    )


@app.post("/analyze/{symbol}", response_model=AnalysisResponse)
async def analyze_symbol(symbol: str, request: AnalysisRequest = None):
    """Run investment analysis on a stock symbol.

    Args:
        symbol: Stock ticker symbol (e.g., AAPL)
        request: Optional request body with analysis parameters

    Returns:
        AnalysisResponse with analysis results
    """
    if Agent is None:
        raise HTTPException(
            status_code=503,
            detail="victor-core not installed. Install with: pip install victor wheel",
        )

    # Use request params or defaults
    mode_str = request.mode if request else "standard"
    try:
        analysis_mode = AnalysisMode(mode_str)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode: {mode_str}. Use: quick, standard, comprehensive",
        )

    try:
        # Build workflow
        workflow = build_graph_for_mode(analysis_mode)

        # Execute
        initial_state = {
            "symbol": symbol.upper(),
            "mode": analysis_mode,
        }
        result = await workflow.invoke(initial_state)

        # Convert to response
        return AnalysisResponse(
            symbol=symbol.upper(),
            mode=mode_str,
            status="completed",
            fundamental_analysis=result.fundamental_analysis,
            technical_analysis=result.technical_analysis,
            market_context=result.market_context,
            synthesis=result.synthesis,
            recommendation=result.recommendation,
            errors=result.errors,
            timestamp=datetime.utcnow().isoformat(),
        )

    except Exception as e:
        logger.error(f"Analysis failed for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch", response_model=BatchAnalysisResponse)
async def batch_analyze(
    request: BatchAnalysisRequest,
    background_tasks: BackgroundTasks,
):
    """Submit batch analysis for multiple symbols.

    Args:
        request: Batch analysis request with list of symbols

    Returns:
        Job ID for tracking batch progress
    """
    if not request.symbols:
        raise HTTPException(status_code=400, detail="No symbols provided")

    # Generate job ID
    job_id = f"batch_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    # Store job
    app.state.analysis_jobs[job_id] = {
        "symbols": request.symbols,
        "mode": request.mode,
        "status": "pending",
        "results": {},
        "submitted_at": datetime.utcnow().isoformat(),
    }

    # Add background task
    background_tasks.add_task(_run_batch_analysis, job_id, request.symbols, request.mode)

    return BatchAnalysisResponse(
        submitted=len(request.symbols),
        job_id=job_id,
        status="pending",
    )


@app.get("/batch/{job_id}")
async def get_batch_status(job_id: str):
    """Get status of a batch analysis job."""
    if job_id not in app.state.analysis_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = app.state.analysis_jobs[job_id]

    # Calculate progress
    total = len(job["symbols"])
    completed = len(job["results"])
    progress = completed / total if total > 0 else 0

    return {
        **job,
        "progress": progress,
        "completed_count": completed,
        "total_count": total,
    }


@app.get("/models")
async def list_models():
    """List available Ollama models."""
    try:
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:11434/api/tags") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    models = data.get("models", [])
                    return {
                        "models": [
                            {
                                "name": m.get("name"),
                                "size": m.get("size"),
                                "modified": m.get("modified_at"),
                            }
                            for m in models
                        ],
                        "count": len(models),
                    }
                else:
                    raise HTTPException(status_code=resp.status, detail="Ollama unavailable")
    except aiohttp.ClientError as e:
        raise HTTPException(status_code=503, detail=f"Cannot connect to Ollama: {e}")


@app.get("/cache/stats")
async def cache_stats():
    """Get cache statistics."""
    try:
        if app.state.cache_manager:
            stats = app.state.cache_manager.get_stats()
            return {"status": "ok", "stats": stats}
        else:
            # Try to get stats directly
            from investigator.infrastructure.cache import CacheManager

            cache = CacheManager()
            stats = cache.get_stats()
            return {"status": "ok", "stats": stats}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/cache/warm")
async def warm_cache(request: CacheWarmRequest, background_tasks: BackgroundTasks):
    """Warm cache for specified symbols."""
    try:
        if not request.symbols:
            raise HTTPException(status_code=400, detail="No symbols provided")

        # Add background task
        background_tasks.add_task(_warm_cache_for_symbols, request.symbols)

        return {
            "message": f"Cache warming started for {len(request.symbols)} symbols",
            "symbols": request.symbols,
            "status": "started",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/cache/symbol/{symbol}")
async def clear_symbol_cache(symbol: str):
    """Clear cache for a specific symbol."""
    try:
        symbol = symbol.upper()

        # Try to clear from cache manager
        if app.state.cache_manager and hasattr(app.state.cache_manager, "clear_symbol"):
            await app.state.cache_manager.clear_symbol(symbol)
        else:
            # Try direct approach
            from investigator.infrastructure.cache import CacheManager

            cache = CacheManager()
            if hasattr(cache, "clear_symbol"):
                cache.clear_symbol(symbol)

        return {
            "message": f"Cache cleared for {symbol}",
            "status": "success",
        }
    except Exception as e:
        return {
            "message": f"Cache clear attempted for {symbol}",
            "status": "partial",
            "error": str(e),
        }


# ========================================================================================
# Background Tasks
# ========================================================================================


async def _run_batch_analysis(job_id: str, symbols: List[str], mode: str):
    """Background task for batch analysis."""
    app.state.analysis_jobs[job_id]["status"] = "running"

    analysis_mode = AnalysisMode(mode)
    workflow = build_graph_for_mode(analysis_mode)

    for symbol in symbols:
        try:
            initial_state = {"symbol": symbol.upper(), "mode": analysis_mode}
            result = await workflow.invoke(initial_state)

            app.state.analysis_jobs[job_id]["results"][symbol] = {
                "status": "completed",
                "recommendation": result.recommendation,
                "errors": result.errors,
            }
        except Exception as e:
            app.state.analysis_jobs[job_id]["results"][symbol] = {
                "status": "error",
                "error": str(e),
            }

    app.state.analysis_jobs[job_id]["status"] = "completed"
    app.state.analysis_jobs[job_id]["completed_at"] = datetime.utcnow().isoformat()


async def _warm_cache_for_symbols(symbols: List[str]):
    """Background task for cache warming."""
    try:
        from victor_invest.tools import MarketDataTool, SECFilingTool

        sec_tool = SECFilingTool()
        market_tool = MarketDataTool()

        for symbol in symbols:
            try:
                # Fetch SEC data to warm cache
                await sec_tool.execute(symbol=symbol.upper())
                # Fetch market data to warm cache
                await market_tool.execute(symbol=symbol.upper(), action="get_price_history")
                logger.info(f"Cache warmed for {symbol}")
            except Exception as e:
                logger.warning(f"Failed to warm cache for {symbol}: {e}")
    except Exception as e:
        logger.error(f"Cache warming failed: {e}")


# ========================================================================================
# Error Handlers
# ========================================================================================


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


# ========================================================================================
# Main Entry Point
# ========================================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "victor_invest.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
