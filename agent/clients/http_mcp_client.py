import logging
from typing import Optional, Any

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import CallToolResult, TextContent

logger = logging.getLogger(__name__)


class HttpMCPClient:
    """Handles MCP server connection and tool execution"""

    def __init__(self, mcp_server_url: str) -> None:
        self.server_url = mcp_server_url
        self.session: Optional[ClientSession] = None
        self._streams_context = None
        self._session_context = None
        logger.debug("HttpMCPClient instance created", extra={"server_url": mcp_server_url})

    @classmethod
    async def create(cls, mcp_server_url: str) -> 'HttpMCPClient':
        """Async factory method to create and connect MCPClient"""
        
        # 1. Create instance `cls(mcp_server_url)`
        instance = cls(mcp_server_url)
        # 2. Connect to MCP Server (method `connect`)
        await instance.connect()
        # 3. Return created instance
        return instance

    async def connect(self):
        """Connect to MCP server"""

        # 1. Set `self._streams_context` as `streamablehttp_client(self.server_url)`
        self._streams_context = streamablehttp_client(self.server_url)
        
        # 2. Create `read_stream, write_stream, _` variables from result if execution of `await self._streams_context.__aenter__()`
        read_stream, write_stream, _ = await self._streams_context.__aenter__()
        
        # 3. Set `self._session_context` as `ClientSession(read_stream, write_stream)`
        self._session_context = ClientSession(read_stream, write_stream)
        
        # 4. Set `self.session: ClientSession` as `await self._session_context.__aenter__()`
        self.session = await self._session_context.__aenter__()
        
        # 5. Call session initialization (initialize method) and assign results to `init_result` variable (initialize is async)
        init_result = await self.session.initialize()
        
        # 6. Log the `init_result` to see in logs MCP server capabilities
        logger.info("MCP session initialized", extra={"init_result": init_result})

    async def get_tools(self) -> list[dict[str, Any]]:
        """Get available tools from MCP server"""

        # 1. Check if session is present, if not then raise an error with message that MCP client is not connected to MCP server
        # 2. Through the session get list tools (it is async method, await it)
        # 3. Retrieved tools are returned according MCP (Anthropic) spec. You need to covert it to the DIAL (OpenAI compatible)
        #    tool format https://dialx.ai/dial_api#operation/sendChatCompletionRequest (see tools param)
        # 4. Log retrieved tools
        # 5. Return tools dicts list
        if not self.session:
            logger.error("Attempted to get tools without active session")
            raise RuntimeError("MCP client not connected. Call connect() first.")

        logger.debug("Fetching tools from MCP server", extra={"server_url": self.server_url})        
        tools = await self.session.list_tools()
    

        tool_list = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                }
            }
            for tool in tools.tools
        ]

        logger.info(
            "Retrieved tools from MCP server",
            extra={
                "server_url": self.server_url,
                "tool_count": len(tool_list),
                "tool_names": [tool["function"]["name"] for tool in tool_list]
            }
        )

        return tool_list

    async def call_tool(self, tool_name: str, tool_args: dict[str, Any]) -> Any:
        """Call a specific tool on the MCP server"""

        # 1. Check if session is present, if not then raise an error with message that MCP client is not connected to MCP server
        # 2. Log the call to MCP Server (tool name, tool args, url)
        # 3. Make tool call through session (it is async, don't forget to await)
        # 4. Get tool execution content
        # 5. Get first element from content (it is array with `ContentBlock`)
        # 6. Check if element is instance of TextContent, if yes then return its text, otherwise return retrieved content
        if not self.session:
            logger.error("Attempted to call tool without active session", extra={"tool_name": tool_name})
            raise RuntimeError("MCP client not connected. Call connect() first.")
        
        logger.info(
            "Calling tool on MCP server",
            extra={
                "tool_name": tool_name,
                "tool_args": tool_args,
                "server_url": self.server_url
            }
        )
        result: CallToolResult = await self.session.call_tool(tool_name, tool_args)
        logger.info(
            "Received tool call result from MCP server",
            extra={
                "tool_name": tool_name,
                "result": result
            }
        )
        if result.content and isinstance(result.content[0], TextContent):
            return result.content[0].text
        return result.content   
