# ATLAS Agent AI: Llama 3.1 8B Instruct on Modal

This repository contains the deployment configuration for a high-empathy Public Service Assistant powered by the official **Meta Llama 3.1 8B Instruct** model, running on Modal's serverless GPU infrastructure.

## 🚀 Architecture Overview

The system is optimized for low-latency, empathetic interactions in a public service context.

- **Model**: [Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)
- **Engine**: Hugging Face `transformers` with `bitsandbytes` 4-bit quantization.
- **Infrastructure**: Modal L4 GPU (16GB), Debian Slim Python 3.10.
- **Optimization**: **Baked-in Weights**. The 5GB model is pre-downloaded and snapped into the Modal image during build, ensuring sub-second container startup times.
- **Session State**: Persistent `modal.Dict` used to track user interactions across serverless container restarts.

## 🛠 Setup & Deployment

1. **Hugging Face Access**: Ensure you have accepted the Meta Llama 3.1 license on Hugging Face.
2. **Modal Secrets**:
   - Create `huggingface-secret` containing your `HF_TOKEN`.
   - Create `atlas-api-key` containing your `ATLAS_API_KEY`.

   ```bash
   modal secret create huggingface-secret HF_TOKEN=your_token_here
   modal secret create atlas-api-key ATLAS_API_KEY=your_chosen_secret_key
   ```

3. **Deploy**:

   ```bash
   modal deploy agent_deploy.py
   ```

## 🛡 Persona & Safety Rules

The assistant is configured with a specific persona ("High-empathy Public Service Assistant") and follows strict governance rules:

- **API Security**: The endpoint is secured via an `X-API-Key` header. Requests without a valid key will be rejected.
- **No Person Names**: All references to internal staff names (e.g., "Sarah") have been removed to protect privacy.
- **Gateway Integration**: The agent automatically queries the [ATLAS MCP Gateway](https://atlas-mcp-gateway-vercel.vercel.app/) to evaluate risk and decision labels.
- **EU AI Act**: Explicitly cites Article 14 during human escalations to protect citizens from incorrect automated decisions.

## 🧪 Testing

Test the secured endpoint with `curl` (replace `YOUR_KEY` with the value set in your secret):

```bash
curl -X POST https://aidan-thomas--atlas-agent-ai-atlasagent-chat.modal.run \
     -H "Content-Type: application/json" \
     -H "X-API-Key: YOUR_KEY" \
     -d '{"user_id": "alex_warmup", "message": "Verify system status."}'
```

---
*Architecture and deployment finalized by Nedo for the ATLAS Project.*
