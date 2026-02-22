# ATLAS Agent AI (Modal + Llama 3 LCEL Deployment)

This repository contains the deployment configuration for the **ATLAS Agent AI** endpoint, hosted on [Modal](https://modal.com/) serverless infrastructure.

## Architecture

The original architecture relied on `langchain-ollama` serving an Ollama subprocess. However, due to systemd and PATH restrictions within the Modal container build lifecycle, this was refactored to locally host a quantised Llama 3 model directly in memory using `llama-cpp-python` and huggingface Hub.

### Features

- **LlamaCpp Engine:** Direct fast GGml inference execution of `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` via an L4 GPU slice.
- **LCEL Inference Pipeline:** Replaces the unstable Langchain `AgentExecutor` with a simplified zero-shot PromptTemplate generating structured empathetic outputs.
- **VRAM Cold Start Support:** Explicit 600-second execution timeouts assigned to FastAPI POST routes to support the 5GB initial VRAM model load time.
- **MCP Gateway Integrations:** Natively fetches Atlas Gateway Evaluation definitions (Auto-Approve, Auto-Deny, Escalate to Human).

## Deployment

Deploy the container manually:

```bash
modal deploy agent_deploy.py
```

*Note: Ensure your `modal token new` credentials are authenticated.*

## Web Endpoint Testing

Once deployed to `https://aidan-thomas--atlas-agent-ai-atlasagent-chat.modal.run`, the endpoint is active.

### Test Case 1: Standard Review (Initial State)

This triggers a standard review/escalation pipeline via the MCP gateway payload.

```bash
curl -X POST "https://aidan-thomas--atlas-agent-ai-atlasagent-chat.modal.run" \
     -H "Content-Type: application/json" \
     -d '{
           "user_id": "alex_citizen_001",
           "message": "Hi, I need help reviewing my housing application. I uploaded all the files!"
         }'
```

**Expected Output:** "Your case is currently paused and under review by Case Officer Sarah..."

### Test Case 2: Approval

```bash
curl -X POST "https://aidan-thomas--atlas-agent-ai-atlasagent-chat.modal.run" \
     -H "Content-Type: application/json" \
     -d '{
           "user_id": "alex_citizen_approve001",
           "message": "I uploaded the final signed rental evidence."
         }'
```

### Test Case 3: Auto-Denial

```bash
curl -X POST "https://aidan-thomas--atlas-agent-ai-atlasagent-chat.modal.run" \
     -H "Content-Type: application/json" \
     -d '{
           "user_id": "alex_citizen_deny001",
           "message": "I currently make $250,000 annually and I am seeking low income housing assistance."
         }'
```
