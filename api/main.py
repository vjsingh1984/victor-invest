"""
InvestiGator REST API
FastAPI-based REST API for the agentic investment analysis platform

DEPRECATED: This API is maintained for backwards compatibility only.
New deployments should use the Victor-powered API instead:

    uvicorn victor_invest.api.app:app --port 8000

Or via CLI:

    victor-invest serve --port 8000

The Victor API provides:
- StateGraph-based workflows
- Simpler deployment (no Redis/agent manager required)
- Better health checks
- Multi-provider LLM support

See victor_invest/api/app.py for the new architecture.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any
from datetime import datetime
import asyncio
import logging
import json
import uuid
from enum import Enum

# Import agent components
from agents import create_agent, AGENT_REGISTRY, AgentTask, AnalysisType, Priority, TaskStatus
from agents.manager import AgentManager
from core.ollama_client import OllamaClient
from core.event_bus import EventBus
from core.cache import IntelligentCacheManager
from investigator.infrastructure.database.db import DatabaseManager
from investigator.config import get_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========================================================================================
# Application Lifecycle
# ========================================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - startup and shutdown"""
    # Startup
    logger.info("Starting InvestiGator API...")

    # Load configuration
    config = get_config()
    ollama_base_url = config.ollama.base_url
    logger.info(f"Using Ollama server at: {ollama_base_url}")

    # Initialize core components
    app.state.ollama = OllamaClient(base_url=ollama_base_url)
    app.state.event_bus = EventBus()
    app.state.cache = IntelligentCacheManager()
    app.state.db = DatabaseManager()
    app.state.agent_manager = AgentManager(
        ollama_client=app.state.ollama, event_bus=app.state.event_bus, cache_manager=app.state.cache
    )

    # Connect to services
    await app.state.ollama.__aenter__()
    await app.state.event_bus.connect()
    await app.state.cache.initialize()
    await app.state.db.initialize()

    # Register agents
    for agent_type in AGENT_REGISTRY:
        agent = create_agent(
            agent_type=agent_type,
            agent_id=f"{agent_type}_01",
            ollama_client=app.state.ollama,
            event_bus=app.state.event_bus,
        )
        await app.state.agent_manager.register_agent(agent)
        logger.info(f"Registered agent: {agent_type}")

    # Start background workers
    app.state.background_tasks = []
    app.state.background_tasks.append(asyncio.create_task(app.state.agent_manager.process_queue()))

    logger.info("InvestiGator API started successfully")

    yield

    # Shutdown
    logger.info("Shutting down InvestiGator API...")

    # Cancel background tasks
    for task in app.state.background_tasks:
        task.cancel()

    # Shutdown agents
    await app.state.agent_manager.shutdown()

    # Close connections
    await app.state.ollama.__aexit__(None, None, None)
    await app.state.event_bus.disconnect()
    await app.state.cache.close()
    await app.state.db.close()

    logger.info("InvestiGator API shutdown complete")


# ========================================================================================
# FastAPI Application
# ========================================================================================

app = FastAPI(
    title="InvestiGator API",
    description="AI-powered investment analysis platform using local LLMs",
    version="2.0.0",
    lifespan=lifespan,
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# ========================================================================================
# Request/Response Models
# ========================================================================================


class AnalysisRequest(BaseModel):
    """Request model for analysis endpoints"""

    symbol: str = Field(..., description="Stock symbol to analyze")
    analysis_types: List[str] = Field(
        default=["sec_fundamental", "technical_analysis", "investment_synthesis"],
        description="Types of analysis to perform",
    )
    priority: str = Field(default="medium", description="Task priority")
    options: Dict[str, Any] = Field(default_factory=dict, description="Additional options")

    @validator("symbol")
    def validate_symbol(cls, v):
        if not v or len(v) > 10:
            raise ValueError("Invalid symbol")
        return v.upper()


class AnalysisResponse(BaseModel):
    """Response model for analysis endpoints"""

    workflow_id: str
    symbol: str
    status: str
    message: str
    created_at: datetime
    estimated_completion: Optional[datetime] = None


class WorkflowStatus(BaseModel):
    """Workflow status response"""

    workflow_id: str
    symbol: str
    status: str
    progress: float
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    results: Optional[Dict[str, Any]] = None
    errors: List[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class BatchAnalysisRequest(BaseModel):
    """Request for batch analysis"""

    symbols: List[str] = Field(..., description="List of symbols to analyze")
    analysis_types: List[str] = Field(default=["sec_fundamental", "technical_analysis", "investment_synthesis"])
    priority: str = Field(default="medium")
    parallel: bool = Field(default=True, description="Process symbols in parallel")


class PeerGroupRequest(BaseModel):
    """Request for peer group analysis"""

    symbol: str
    peer_count: int = Field(default=10, ge=5, le=20)
    sector: Optional[str] = None
    industry: Optional[str] = None
    include_comprehensive: bool = Field(default=True)


class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    version: str
    timestamp: datetime
    services: Dict[str, str]
    agents: Dict[str, Dict[str, Any]]


# ========================================================================================
# API Endpoints
# ========================================================================================


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {"name": "InvestiGator API", "version": "2.0.0", "status": "operational", "documentation": "/docs"}


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check(request: Request):
    """Comprehensive health check endpoint"""
    try:
        # Check all services
        services_status = {}

        # Check Ollama
        try:
            models = await request.app.state.ollama.list_models()
            services_status["ollama"] = "healthy" if models else "degraded"
        except:
            services_status["ollama"] = "unhealthy"

        # Check Redis
        try:
            await request.app.state.event_bus.redis_client.ping()
            services_status["redis"] = "healthy"
        except:
            services_status["redis"] = "unhealthy"

        # Check Database
        try:
            await request.app.state.db.health_check()
            services_status["database"] = "healthy"
        except:
            services_status["database"] = "unhealthy"

        # Check Cache
        services_status["cache"] = "healthy"  # Always healthy if initialized

        # Get agent health
        agents_health = await request.app.state.agent_manager.get_agents_health()

        overall_status = "healthy"
        if any(status == "unhealthy" for status in services_status.values()):
            overall_status = "unhealthy"
        elif any(status == "degraded" for status in services_status.values()):
            overall_status = "degraded"

        return HealthResponse(
            status=overall_status,
            version="2.0.0",
            timestamp=datetime.now(),
            services=services_status,
            agents=agents_health,
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


@app.post("/analyze", response_model=AnalysisResponse, tags=["Analysis"])
async def analyze_symbol(request: AnalysisRequest, background_tasks: BackgroundTasks, app_state: Request):
    """Submit a symbol for comprehensive analysis"""
    try:
        # Convert analysis types to enums
        analysis_enums = []
        for analysis_type in request.analysis_types:
            try:
                analysis_enums.append(AnalysisType(analysis_type))
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid analysis type: {analysis_type}")

        # Submit to agent manager
        workflow_id = await app_state.app.state.agent_manager.submit_analysis_request(
            symbol=request.symbol,
            analysis_types=analysis_enums,
            priority=Priority[request.priority.upper()],
            options=request.options,
        )

        # Estimate completion time (rough estimate)
        estimated_completion = datetime.now()
        estimated_completion = estimated_completion.replace(
            minute=estimated_completion.minute + len(request.analysis_types) * 2
        )

        return AnalysisResponse(
            workflow_id=workflow_id,
            symbol=request.symbol,
            status="submitted",
            message=f"Analysis for {request.symbol} has been submitted",
            created_at=datetime.now(),
            estimated_completion=estimated_completion,
        )

    except Exception as e:
        logger.error(f"Analysis submission failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analyze/{workflow_id}", response_model=WorkflowStatus, tags=["Analysis"])
async def get_analysis_status(workflow_id: str, app_state: Request):
    """Get status of an analysis workflow"""
    try:
        status = await app_state.app.state.agent_manager.get_workflow_status(workflow_id)

        if not status:
            raise HTTPException(status_code=404, detail="Workflow not found")

        # Calculate progress
        progress = 0.0
        if status["total_tasks"] > 0:
            progress = (status["completed_tasks"] + status["failed_tasks"]) / status["total_tasks"]

        # Determine overall status
        if status["failed_tasks"] > 0:
            overall_status = "partial_failure"
        elif status["completed_tasks"] == status["total_tasks"]:
            overall_status = "completed"
        else:
            overall_status = "in_progress"

        return WorkflowStatus(
            workflow_id=workflow_id,
            symbol=status.get("symbol", "unknown"),
            status=overall_status,
            progress=progress,
            total_tasks=status["total_tasks"],
            completed_tasks=status["completed_tasks"],
            failed_tasks=status["failed_tasks"],
            results=status.get("results"),
            errors=status.get("errors", []),
            created_at=status.get("created_at", datetime.now()),
            updated_at=datetime.now(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get workflow status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/batch", tags=["Analysis"])
async def analyze_batch(request: BatchAnalysisRequest, background_tasks: BackgroundTasks, app_state: Request):
    """Submit multiple symbols for analysis"""
    try:
        workflows = []

        # Convert analysis types
        analysis_enums = []
        for analysis_type in request.analysis_types:
            try:
                analysis_enums.append(AnalysisType(analysis_type))
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid analysis type: {analysis_type}")

        # Submit each symbol
        for symbol in request.symbols:
            workflow_id = await app_state.app.state.agent_manager.submit_analysis_request(
                symbol=symbol.upper(), analysis_types=analysis_enums, priority=Priority[request.priority.upper()]
            )
            workflows.append({"symbol": symbol.upper(), "workflow_id": workflow_id})

        return {
            "batch_id": str(uuid.uuid4()),
            "symbols_count": len(request.symbols),
            "workflows": workflows,
            "status": "submitted",
            "created_at": datetime.now(),
        }

    except Exception as e:
        logger.error(f"Batch analysis submission failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/peer-group", tags=["Analysis"])
async def analyze_peer_group(request: PeerGroupRequest, app_state: Request):
    """Analyze a symbol with its peer group"""
    try:
        # Get peer group for the symbol
        peers = await app_state.app.state.agent_manager.get_peer_group(
            symbol=request.symbol, sector=request.sector, industry=request.industry, count=request.peer_count
        )

        # Submit analysis for main symbol and peers
        workflows = []

        # Determine analysis types
        if request.include_comprehensive:
            analysis_types = [
                AnalysisType.SEC_FUNDAMENTAL,
                AnalysisType.TECHNICAL_ANALYSIS,
                AnalysisType.PEER_GROUP,
                AnalysisType.INVESTMENT_SYNTHESIS,
            ]
        else:
            analysis_types = [AnalysisType.PEER_GROUP]

        # Submit main symbol
        main_workflow = await app_state.app.state.agent_manager.submit_analysis_request(
            symbol=request.symbol,
            analysis_types=analysis_types,
            priority=Priority.HIGH,
            options={"peer_symbols": peers},
        )

        workflows.append({"symbol": request.symbol, "workflow_id": main_workflow, "is_primary": True})

        # Submit peer analyses
        for peer in peers:
            peer_workflow = await app_state.app.state.agent_manager.submit_analysis_request(
                symbol=peer,
                analysis_types=[AnalysisType.SEC_FUNDAMENTAL, AnalysisType.TECHNICAL_ANALYSIS],
                priority=Priority.MEDIUM,
            )
            workflows.append({"symbol": peer, "workflow_id": peer_workflow, "is_primary": False})

        return {
            "peer_group_id": str(uuid.uuid4()),
            "primary_symbol": request.symbol,
            "peer_symbols": peers,
            "workflows": workflows,
            "status": "submitted",
            "created_at": datetime.now(),
        }

    except Exception as e:
        logger.error(f"Peer group analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models", tags=["Models"])
async def list_models(app_state: Request):
    """List available Ollama models"""
    try:
        models = await app_state.app.state.ollama.list_models()
        return {"models": models, "count": len(models)}
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models/pull", tags=["Models"])
async def pull_model(model_name: str, app_state: Request):
    """Pull a new Ollama model"""
    try:
        # Start pulling model in background
        async def pull_progress():
            async for progress in app_state.app.state.ollama.pull_model(model_name):
                # Could emit progress via WebSocket here
                logger.info(f"Model pull progress: {progress}")

        asyncio.create_task(pull_progress())

        return {"message": f"Started pulling model {model_name}", "status": "in_progress"}
    except Exception as e:
        logger.error(f"Failed to pull model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cache/stats", tags=["Cache"])
async def get_cache_stats(app_state: Request):
    """Get cache statistics"""
    try:
        stats = await app_state.app.state.cache.get_stats()
        return stats
    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/cache/warm", tags=["Cache"])
async def warm_cache(symbols: List[str], app_state: Request):
    """Warm cache for specified symbols"""
    try:
        await app_state.app.state.cache.warm_cache(symbols)
        return {"message": f"Cache warming started for {len(symbols)} symbols", "symbols": symbols}
    except Exception as e:
        logger.error(f"Failed to warm cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/cache/symbol/{symbol}", tags=["Cache"])
async def clear_symbol_cache(symbol: str, app_state: Request):
    """Clear cache for a specific symbol"""
    try:
        await app_state.app.state.cache.clear_symbol(symbol.upper())
        return {"message": f"Cache cleared for {symbol.upper()}", "status": "success"}
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========================================================================================
# WebSocket Endpoints for Real-time Updates
# ========================================================================================

from fastapi import WebSocket, WebSocketDisconnect


@app.websocket("/ws/analysis/{workflow_id}")
async def websocket_analysis_updates(websocket: WebSocket, workflow_id: str, app_state: Request):
    """WebSocket endpoint for real-time analysis updates"""
    await websocket.accept()

    try:
        # Subscribe to workflow events
        async def send_updates():
            async for event in app_state.app.state.event_bus.subscribe_to_workflow(workflow_id):
                await websocket.send_json({"type": "update", "workflow_id": workflow_id, "event": event.dict()})

        await send_updates()

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for workflow {workflow_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()


@app.websocket("/ws/events")
async def websocket_events(websocket: WebSocket, app_state: Request):
    """WebSocket endpoint for all system events"""
    await websocket.accept()

    try:
        # Subscribe to all events
        async for event in app_state.app.state.event_bus.listen_for_events():
            await websocket.send_json({"type": "event", "data": event.dict()})

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for events")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()


# ========================================================================================
# Error Handlers
# ========================================================================================


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code, "timestamp": datetime.now().isoformat()},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "status_code": 500, "timestamp": datetime.now().isoformat()},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
