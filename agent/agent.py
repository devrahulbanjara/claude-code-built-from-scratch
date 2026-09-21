from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Self

from agent.events import AgentEvent, AgentEventType
from client.llm_client import LLMClient
from client.response import StreamEventType
from context.manager import ContextManager
from tools.registry import create_default_registry


class Agent:
    def __init__(self):
        self.client = LLMClient()
        self.context_manager = ContextManager()
        self.tool_registry = create_default_registry()

    async def run(self, message: str):
        yield AgentEvent.agent_start(message)

        self.context_manager.add_user_message(content=message)

        final_response: str | None = None

        async for event in self._agentic_loop(message):
            yield event

            if event.type == AgentEventType.TEXT_COMPLETE:
                final_response = event.data.get("content")

        yield AgentEvent.agent_end(final_response)

    async def _agentic_loop(self, message: str) -> AsyncGenerator[AgentEvent]:
        response_text = ""

        tool_schemas = self.tool_registry.get_schemas()

        async for event in self.client.chat_completions(
            messages=self.context_manager.get_messages(),
            tools=tool_schemas if tool_schemas else None,
        ):
            if event.type == StreamEventType.TEXT_DELTA:
                if event.text_delta:
                    content = event.text_delta.content
                    response_text += content
                    yield AgentEvent.text_delta(content=content)
            elif event.type == StreamEventType.ERROR:
                yield AgentEvent.agent_error(event.error or "Unknown error occurred")

        self.context_manager.add_assistant_message(content=response_text or None)

        if response_text:
            yield AgentEvent.text_complete(response_text)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.client:
            await self.client.close()
            self.client = None
