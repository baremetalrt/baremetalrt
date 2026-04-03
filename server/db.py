"""Async Postgres connection pool using asyncpg."""

import asyncpg
from server.config import DATABASE_URL

_pool: asyncpg.Pool | None = None


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10)
    return _pool


async def close_pool():
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


async def fetch_one(query: str, *args):
    pool = await get_pool()
    return await pool.fetchrow(query, *args)


async def fetch_all(query: str, *args):
    pool = await get_pool()
    return await pool.fetch(query, *args)


async def execute(query: str, *args):
    pool = await get_pool()
    return await pool.execute(query, *args)
