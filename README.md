# profile-rag
End-to-End RAG pipeline that scrapes and retrieves professional profile data using MCP servers, vector databases, and LLMs.

---
Workflow structure:

## The order matters for debugging
```
settings.py         ← fix first if imports break
    ↓
llm_interface.py    ← fix if Groq auth fails
    ↓
mcp_server/server.py ← fix if scraping fails
    ↓
data_processing.py  ← fix if chunking/embedding fails
    ↓
query_engine.py     ← fix if LLM responses are wrong
    ↓
profile_service.py  ← fix if orchestration breaks
    ↓
api/app.py          ← fix if HTTP endpoints fail
    ↓
main.py             ← fix last, only if UI/launch issues
Always test each layer independently before moving to the next. What layer do you want to start with?