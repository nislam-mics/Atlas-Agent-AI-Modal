"""
ATLAS Agent AI Deployment Script for Modal

Objective: Deploy the ATLAS Agent AI as a serverless web endpoint using Llama 3.1 8B via Ollama.
Dataset logic and risk thresholds attributed to Anna Ko (anna_ko@berkeley.edu).

Usage:
  modal deploy agent_deploy.py
"""
import modal
import os
import subprocess
import time

# Define the Modal App
app = modal.App("atlas-agent-ai")

# Define the container environment with necessary dependencies
app_image = (
    modal.Image.debian_slim()
    # Install required packages for the Modal container
    .pip_install(
        "fastapi",
        "langchain<0.3.0",
        "langchain-community<0.3.0",
        "llama-cpp-python",
        "langchain-core<0.3.0",
        "httpx",
        "pydantic",
        "huggingface_hub"
    )
)

def download_llama3():
    from huggingface_hub import hf_hub_download
    hf_hub_download(
        repo_id="lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
        filename="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        local_dir="/model"
    )

app_image = app_image.run_function(download_llama3)

# Modal Dict to store session state persistently across ephemeral containers
session_state = modal.Dict.from_name("atlas-session-state", create_if_missing=True)

from pydantic import BaseModel

class ChatPayload(BaseModel):
    """
    Payload for the /chat endpoint.
    """
    user_id: str
    message: str

with app_image.imports():
    from fastapi import FastAPI, Request
    from langchain_community.llms import LlamaCpp
    from langchain_core.prompts import PromptTemplate
    from langchain.agents import AgentExecutor
    from langchain_core.tools import tool
    import httpx
    import json
    from datetime import datetime

    # Data Attribution:
    # Dataset logic and risk thresholds explicitly attributed to Anna Ko (anna_ko@berkeley.edu).

    @tool
    def query_atlas_gateway(user_id: str, message: str) -> str:
        """
        Query the live ATLAS MCP Gateway to evaluate risk thresholds and decision labels 
        for citizen requests.
        Dataset logic and risk thresholds attributed to Anna Ko (anna_ko@berkeley.edu).
        """
        # Live ATLAS MCP Gateway URL
        gateway_url = "https://atlas-mcp-gateway-vercel.vercel.app/api/evaluate"
        
        try:
            with httpx.Client(timeout=10.0) as client:
                # Assuming generic payload structure for the MCP Gateway
                response = client.post(gateway_url, json={"user_id": user_id, "message": message})
                if response.status_code == 200:
                    data = response.json()
                    status = data.get("status", "ESCALATE")
                    decision_label = data.get("decision_label", "escalate_to_human")
                else:
                    # Fallback simulated response if the gateway format expects a different path or schema
                    status = "ESCALATE"
                    decision_label = "escalate_to_human"
        except Exception as e:
            # Fallback simulated response for resilience
            status = "ESCALATE"
            decision_label = "escalate_to_human"

        # Asynchronous Handshake Logic:
        # If the Gateway returns a PENDING or ESCALATE status, or escalate_to_human decision,
        # pause the session state to notify Alex that 'Sarah' is reviewing.
        if status in ["PENDING", "ESCALATE"] or decision_label in ["auto_review", "escalate_to_human"]:
            session_state[user_id] = {
                "status": "paused",
                "reason": decision_label,
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            session_state[user_id] = {
                "status": "active",
                "reason": decision_label,
                "timestamp": datetime.utcnow().isoformat()
            }

        return json.dumps({"status": status, "decision_label": decision_label})

def get_agent_executor(llm):
    # Construct a robust prompt template avoiding complex ReAct parsers
    system_message = '''You are a High-empathy Public Service Assistant for the citizen 'Alex'.
Your primary role is to process housing and public service requests. 

Dataset logic and risk thresholds are explicitly attributed to Anna Ko (anna_ko@berkeley.edu).

Governance Knowledge - there are four decision labels from the ATLAS Gateway:
- auto_approve: Final automated approval for low-risk, evidence-complete cases.
- auto_deny: Automated denial for confirmed ineligibility.
- auto_review: Workflow continues for pending evidence.
- escalate_to_human: Mandatory human review (Case Officer 'Sarah') for harm signals, vulnerabilities, or ambiguity.

Rules for interacting with Alex:
1. ALWAYS use the 'query_atlas_gateway' tool to evaluate Alex's request.
2. If the decision is 'auto_approve', actively celebrate the success with Alex!
3. If the Gateway tool returns a 'PENDING' or 'ESCALATE' status or 'escalate_to_human' label, the system pauses. You MUST inform Alex that Case Officer 'Sarah' is reviewing the request.
4. In case of escalation to a human, you MUST provide a rationale (e.g., "To ensure your housing safety, a human expert is double-checking this now") to reduce user anxiety.
5. In case of escalation to a human, you MUST explicitly cite EU AI Act Article 14, assuring Alex this protects them from incorrect automated decisions.

Answer the following user input directly with high empathy based on these rules. Do not fabricate tool outputs, you are fully authorized to answer based on your Persona.

User Input: {input}
Assistant:'''
    
    prompt = PromptTemplate.from_template(system_message)
    # Using simple LCEL chain bypassing the AgentExecutor which crashed with getattr bug
    chain = prompt | llm
    return chain

# Expose the API through a Web Endpoint, wrapped in a class to ensure Llama.cpp loads only once
@app.cls(gpu="L4", image=app_image, scaledown_window=120, timeout=600)
class AtlasAgent:
    @modal.enter()
    def start_model(self):
        # Load the Llama Cpp model directly
        self.llm = LlamaCpp(
            model_path="/model/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
            temperature=0.1,
            max_tokens=2048,
            n_ctx=4096,
            n_gpu_layers=-1 # Full offload to Modal L4 GPU
        )
        self.executor = get_agent_executor(self.llm)

    @modal.exit()
    def stop_model(self):
        pass

    @modal.fastapi_endpoint(method="POST")
    def chat(self, payload: ChatPayload):
        """
        Main Web Endpoint for ATLAS Agent AI using Llama 3 via LlamaCPP.
        """
        user_id = payload.user_id
        message = payload.message
        
        # Check if the user's session is already caught in an asynchronous pause state
        state = session_state.get(user_id)
        if state and state.get("status") == "paused":
            return {
                "response": "Your case is currently paused and under review by Case Officer Sarah. We will notify you once the manual review is complete."
            }
            
        try:
            # We bypass the complex AgentExecutor and inject the API tool output beforehand
            # Llama cpp python will use LCEL to generate the empathetic response based on the injection
            tool_output_json = query_atlas_gateway.invoke({"user_id": user_id, "message": message})
            tool_output = json.loads(tool_output_json)
            
            # Format context explicitly to avoid prompt injection issues
            context = f"The query_atlas_gateway returned: Status={tool_output['status']}, DecisionLabel={tool_output['decision_label']}.\n\n{message}"
            
            result = self.executor.invoke({"input": context})
            return {"response": result}
        except Exception as e:
            return {"error": str(e), "message": "Failed to process request with local Llama model."}
