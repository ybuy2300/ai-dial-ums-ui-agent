```bash
uvicorn agent.app:app --host 0.0.0.0 --port 8011 --log-level debug --reload
```

```text
INFO:     Will watch for changes in these directories: ['/Users/Yuriy_Buy/IdeaProjects/DAIL/ai-dial-ums-ui-agent']
INFO:     Uvicorn running on http://0.0.0.0:8011 (Press CTRL+C to quit)
INFO:     Started reloader process [72869] using StatReload
INFO:     Started server process [72873]
INFO:     Waiting for application startup.
2026-02-11 18:22:40,309 - agent.app - INFO - Application startup initiated
2026-02-11 18:22:40,309 - agent.app - INFO - Initializing UMS MCP client
2026-02-11 18:22:40,309 - agent.app - INFO - UMS MCP URL: http://192.168.1.24:8005/mcp
2026-02-11 18:22:40,309 - agent.clients.http_mcp_client - DEBUG - HttpMCPClient instance created
2026-02-11 18:22:40,325 - mcp.client.streamable_http - DEBUG - Connecting to StreamableHTTP endpoint: http://192.168.1.24:8005/mcp
2026-02-11 18:22:40,325 - mcp.client.streamable_http - DEBUG - Sending client message: root=JSONRPCRequest(method='initialize', params={'protocolVersion': '2025-11-25', 'capabilities': {}, 'clientInfo': {'name': 'mcp', 'version': '0.1.0'}}, jsonrpc='2.0', id=0)
2026-02-11 18:22:40,325 - httpcore.connection - DEBUG - connect_tcp.started host='192.168.1.24' port=8005 local_address=None timeout=30 socket_options=None
```

Open index.html in the browser