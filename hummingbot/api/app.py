"""
FastAPI Application Setup
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from hummingbot.api.routers import connectors, controllers, strategies, system, trading

logger = logging.getLogger(__name__)

# Global reference to the running server
_server: Optional[uvicorn.Server] = None
_server_task: Optional[asyncio.Task] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events"""
    logger.info("Hummingbot API server starting...")
    yield
    logger.info("Hummingbot API server shutting down...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    app = FastAPI(
        title="Hummingbot API",
        description="REST API for controlling Hummingbot trading bot",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(system.router, prefix="/api/v1", tags=["System"])
    app.include_router(connectors.router, prefix="/api/v1/connectors", tags=["Connectors"])
    app.include_router(controllers.router, prefix="/api/v1/controllers", tags=["Controllers"])
    app.include_router(strategies.router, prefix="/api/v1/strategies", tags=["Strategies"])
    app.include_router(trading.router, prefix="/api/v1/trading", tags=["Trading"])

    return app


async def start_api_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Start the API server in the background"""
    global _server, _server_task

    if _server_task is not None:
        logger.warning("API server is already running")
        return

    app = create_app()
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="warning",
        access_log=False,
    )
    _server = uvicorn.Server(config)

    _server_task = asyncio.create_task(_server.serve())
    logger.info(f"Hummingbot API server started at http://{host}:{port}")
    logger.info(f"API documentation available at http://{host}:{port}/docs")


async def stop_api_server() -> None:
    """Stop the API server"""
    global _server, _server_task

    if _server is not None:
        _server.should_exit = True
        if _server_task is not None:
            try:
                await asyncio.wait_for(_server_task, timeout=5.0)
            except asyncio.TimeoutError:
                _server_task.cancel()
            _server_task = None
        _server = None
        logger.info("Hummingbot API server stopped")
