import json
import logging
import os
import uuid
from datetime import datetime, UTC
from typing import Optional, AsyncGenerator

import redis.asyncio as redis

from agent.clients.dial_client import DialClient
from agent.models.message import Message, Role
from agent.prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

CONVERSATION_PREFIX = "conversation:"
CONVERSATION_LIST_KEY = "conversations:list"


class ConversationManager:
    """Manages conversation lifecycle including AI interactions and persistence"""

    def __init__(self, dial_client: DialClient, redis_client: redis.Redis):
        self.dial_client = dial_client
        self.redis = redis_client
        logger.info("ConversationManager initialized")

    async def create_conversation(self, title: str) -> dict:
        """Create a new conversation"""

        # 1. Create conversation id `str(uuid.uuid4())`
        conversation_id = str(uuid.uuid4())
        
        # 2. Create current datatime `datetime.now(UTC).isoformat()`
        current_datetime = datetime.now(UTC).isoformat()
        
        # 3. Create `conversation` dict with:
        #       - "id": conversation_id
        #       - "title": title
        #       - "messages": []
        #       - "created_at": created datatime from 2nd point
        #       - "updated_at": created datatime from 2nd point
        conversation = {
            "id": conversation_id,
            "title": title,
            "messages": [],
            "created_at": current_datetime,
            "updated_at": current_datetime
        }

        # 4. Set conversation in redis (`set` is async, don't forget to await) with:
        #       - f"{CONVERSATION_PREFIX}{conversation_id}"
        #       - json.dumps(conversation)
        await self.redis.set(f"{CONVERSATION_PREFIX}{conversation_id}", json.dumps(conversation))

        # 5. Add conversation in redis (`zadd` is async, don't forget to await) with:
        #       - CONVERSATION_LIST_KEY
        #       - {conversation_id: datetime.now(UTC).timestamp()}
        await self.redis.zadd(CONVERSATION_LIST_KEY, {conversation_id: datetime.now(UTC).timestamp()})

        # 6. Log the conversation info
        logger.info("Created new conversation", extra={"conversation_id": conversation_id, "title": title})

        # 7. Return conversation
        return conversation


    async def list_conversations(self) -> list[dict]:
        """List all conversations sorted by last update time"""

        logger.debug("Listing all conversations")
        # 1. Get `conversation_ids` with `await self.redis.zrevrange(CONVERSATION_LIST_KEY, 0, -1)`
        conversation_ids = await self.redis.zrevrange(CONVERSATION_LIST_KEY, 0, -1)
        # 2. Create empty list as `conversations`
        conversations = []
        
        # 3. Iterate through `conversation_ids` and:
        #       - get conversation from redis, use CONVERSATION_PREFIX before conversation_id (don't forget to await, it is async)
        #       - if conversation is present then:
        #           - load it with json (json.loads)
        #           - add to `conversations` list a dict with:
        #               - id": conv["id"]
        #               - "title": conv["title"]
        #               - "created_at": conv["created_at"]
        #               - "updated_at": conv["updated_at"]
        #               - "message_count": len(conv["messages"])
        for conversation_id in conversation_ids:
            conv_data = await self.redis.get(f"{CONVERSATION_PREFIX}{conversation_id}")
            if conv_data:
                conv = json.loads(conv_data)
                conversations.append({
                    "id": conv["id"],
                    "title": conv["title"],
                    "created_at": conv["created_at"],
                    "updated_at": conv["updated_at"],
                    "message_count": len(conv["messages"])
                })

        # 4. return conversations
        return conversations

    async def get_conversation(self, conversation_id: str) -> Optional[dict]:
        """Get a specific conversation"""
        logger.debug("Retrieving conversation", extra={"conversation_id": conversation_id})

        # 1. Get conversation from redis, use CONVERSATION_PREFIX before conversation_id (don't forget to await, it is async)
        conv_data = await self.redis.get(f"{CONVERSATION_PREFIX}{conversation_id}")
        
        # 2. If nothing found then return None
        if not conv_data:
            return None
        
        # 3. Load it with json (json.loads)
        conversation = json.loads(conv_data)
        logger.debug(
            "Conversation retrieved",
            extra={
                "conversation_id": conversation_id,
                "message_count": len(conversation.get("messages", []))
            }
        )

        # 4. return conversation
        return conversation

    async def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation"""
        logger.info("Deleting conversation", extra={"conversation_id": conversation_id})

        # 1. Call delete conversation in redis, use CONVERSATION_PREFIX before conversation_id (don't forget to await, it is async)
        deleted = await self.redis.delete(f"{CONVERSATION_PREFIX}{conversation_id}")
        # 2. Id nothing was deleted then return False, otherwise True
        if deleted == 0:
            logger.warning("Conversation not found for deletion", extra={"conversation_id": conversation_id})
            return False

        await self.redis.zrem(CONVERSATION_LIST_KEY, conversation_id)
        logger.info("Conversation deleted successfully", extra={"conversation_id": conversation_id})

        return True

    async def chat(
            self,
            user_message: Message,
            conversation_id: str,
            stream: bool = False
    ):
        """
        Process chat messages and return AI response.
        Automatically saves conversation state.
        """

        # 1. Log request
        logger.info(
            "Received chat request",
            extra={"conversation_id": conversation_id, "user_message": user_message.content, "stream": stream})
        
        # 2. Get conversation (use method `get_conversation`)
        conversation = await self.get_conversation(conversation_id)
        
        # 3. Raise an error that no conversation foud if conversation is not present
        if not conversation:
            logger.error("Conversation not found", extra={"conversation_id": conversation_id})
            raise ValueError("No conversation found")
        
        # 4. Get `messages` from conversation, iterate through them and create array with `Message(**msg_data)`
        messages = [Message(**msg_data) for msg_data in conversation.get("messages", [])]
        logger.debug(
            "Loaded conversation history",
            extra={
                "conversation_id": conversation_id,
                "message_count": len(messages)
            }
        )

        # 5. If `messages` array is empty it means that it is beginning of the conversation. Add system prompt as 1st message
        if not messages:
            messages.append(Message(role="system", content=SYSTEM_PROMPT))
        
        # 6. Add `user_message` to `messages` array
        messages.append(user_message)
        
        # 7. If `stream` is true then call `_stream_chat` (without await!), otherwise call `_non_stream_chat` (with await) and return it
        if stream:
            return self._stream_chat(conversation_id, messages)
        else:
            return await self._non_stream_chat(conversation_id, messages)


    async def _stream_chat(
            self,
            conversation_id: str,
            messages: list[Message],
    ) -> AsyncGenerator[str, None]:
        """Handle streaming chat with automatic saving"""
        logger.debug("Starting streaming chat", extra={"conversation_id": conversation_id})

        # 1. Send conversation_id first: `yield f"data: {json.dumps({'conversation_id': conversation_id})}\n\n"`
        yield f"data: {json.dumps({'conversation_id': conversation_id})}\n\n"
        
        # 2. Stream the response - full_messages will be modified by dial_client:
        #       `async for chunk in self.dial_client.stream_response(messages):
        #           yield chunk`
        async for chunk in self.dial_client.stream_response(messages):
            yield chunk
        
        # 3. Save conversation (`_save_conversation_messages` method, don't forget to await)
        await self._save_conversation_messages(conversation_id, messages)
        logger.info(
            "Streaming chat completed is finished",
            extra={ "conversation_id": conversation_id}
        )        

    async def _non_stream_chat(
            self,
            conversation_id: str,
            messages: list[Message],
    ) -> dict:
        """Handle non-streaming chat"""
        logger.debug("Starting non-streaming chat", extra={"conversation_id": conversation_id})

        # 1. Call `await self.dial_client.response(messages)`
        ai_message = await self.dial_client.response(messages)
        
        # 2. Save conversation (`_save_conversation_messages` method, don't forget to await)
        await self._save_conversation_messages(conversation_id, messages + [ai_message])
        
        logger.info(
            "Non-streaming chat completed",
            extra={"conversation_id": conversation_id}
        )

        # 3. Return dict with:
        #       - "content": ai_message.content or ''
        #       - "conversation_id": conversation_id
        return {"content": ai_message.content or '', "conversation_id": conversation_id}

    async def _save_conversation_messages(
            self,
            conversation_id: str,
            messages: list[Message]
    ):
        """Save or update conversation messages"""
        logger.debug(
            "Saving conversation messages",
            extra={ "conversation_id": conversation_id}
        )

        # 1. Get conversation from redis, use CONVERSATION_PREFIX before conversation_id (don't forget to await, it is async)
        conv_data = await self.redis.get(f"{CONVERSATION_PREFIX}{conversation_id}")
        
        # 2. Load it with json (json.loads) as `conversation`
        conversation = json.loads(conv_data)
        
        # 3. Create list with messages dits (use `model_dump` method) and it set `conversation` 'messages'
        conversation["messages"] = [msg.model_dump() for msg in messages]

        # 4. Update `updated_at` time with `datetime.now(UTC).isoformat()` in `conversation`
        conversation["updated_at"] = datetime.now(UTC).isoformat()

        logger.debug("Updating existing conversation", extra={"conversation_id": conversation_id})

        # 5. Save it with `_save_conversation` method
        await self._save_conversation(conversation)

    async def _save_conversation(self, conversation: dict):
        """Internal method to persist conversation to Redis"""

        # 1. Get conversation id
        conversation_id = conversation["id"]
        
        # 2. Call redis set with:
        #       - f"{CONVERSATION_PREFIX}{conversation_id}"
        #       - json.dumps(conversation)
        await self.redis.set(f"{CONVERSATION_PREFIX}{conversation_id}", json.dumps(conversation))
        
        # 3. Call redis zadd with:
        #       - CONVERSATION_LIST_KEY
        #       - {conversation_id: datetime.now(UTC).timestamp()}
        await self.redis.zadd(CONVERSATION_LIST_KEY, {conversation_id: datetime.now(UTC).timestamp()})

        logger.debug(
            "Conversation persisted to Redis",
            extra={
                "conversation_id": conversation_id,
                "message_count": len(conversation.get("messages", []))
            }
        )        

