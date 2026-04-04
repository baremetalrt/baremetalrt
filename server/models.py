"""Pydantic models for request/response validation."""

from pydantic import BaseModel
from typing import Optional


# Auth
class UserResponse(BaseModel):
    id: str
    email: str
    name: Optional[str] = None
    avatar_url: Optional[str] = None
    provider: str
    is_admin: bool = False


# API Keys
class CreateKeyRequest(BaseModel):
    name: str = "default"
    scopes: list[str] = ["inference", "mesh"]


class ApiKeyResponse(BaseModel):
    id: str
    key_prefix: str
    name: str
    scopes: list[str]
    created_at: str
    last_used: Optional[str] = None
    revoked: bool = False


class ApiKeyCreated(ApiKeyResponse):
    key: str  # full key, shown only once


# Nodes
class NodeRegisterRequest(BaseModel):
    node_id: str
    hostname: str
    lan_ip: str
    port: int = 8080
    gpu_name: str
    gpu_vram_total_mb: int
    engine_name: Optional[str] = None
    available_ranks: list[int] = [0]


class NodeResponse(BaseModel):
    node_id: str
    hostname: str
    gpu_name: str
    gpu_vram_mb: int
    status: str
    engine_name: Optional[str] = None


# Chat
class ChatRequest(BaseModel):
    message: str
    max_tokens: int = 2048
    history: list[dict] = []
    model: Optional[str] = None
