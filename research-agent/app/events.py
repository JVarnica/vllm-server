"""Per-task event emission. Events to Redis List (events:{task_id}) which android UI polls """
import logging
import json
import time
from typing import Any
from dataclasses import dataclass, field

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

# set in lifecycle
redis: aioredis.Redis | None = None
TASK_TTL = 3600 # 1 HOUR    

class TaskChannel:
  
    def __init__(self, task_id: str):
        self.task_id = task_id
    
    async def emit(self, event_type: str, data: dict):
        assert redis is not None, "Redis client not initialized"
        key = f"events:{self.task_id}"
        payload = json.dumps({"type": event_type, "ts": time.time(), **data})
        await redis.rpush(key, payload)
        await redis.expire(key, TASK_TTL)



def get_channel(task_id: str) -> TaskChannel:
    return TaskChannel(task_id)


def remove_channel(task_id: str) -> None:
    pass