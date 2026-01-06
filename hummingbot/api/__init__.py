"""
Hummingbot Embedded REST API

A FastAPI-based REST API that provides programmatic access to Hummingbot functionality.
"""

from hummingbot.api.app import create_app, start_api_server, stop_api_server

__all__ = ["create_app", "start_api_server", "stop_api_server"]
