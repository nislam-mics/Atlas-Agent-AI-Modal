import modal
import json
import os
import httpx
from datetime import datetime
from pydantic import BaseModel
from typing import Optional
from fastapi import Header, HTTPException

# Setup the Modal App
app = modal.App("atlas-agent-ai")

# Persistent state across containers
session_state = modal.Dict.from_name("atlas-session-state", create_if_missing=True)

def download_model_to_image():
    """
    Helper function to download the official Llama 3.1 8B Instruct weights
    during the Modal image build process.
    """
    from huggingface_hub import snapshot_download
    snapshot_download(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        local_dir="/model",
        ignore_patterns=["*.pth", "*.pt"]
    )

# Define the container environment with necessary dependencies for Transformers
app_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "fastapi",
        "langchain<0.3.0",
        "langchain-community<0.3.0",
        "langchain-core<0.3.0",
        "httpx",
        "pydantic",
        "transformers",
        "torch",
        "accelerate",
        "bitsandbytes",
        "sentencepiece",
        "huggingface_hub"
    )
    .run_function(
        download_model_to_image,
        secrets=[modal.Secret.from_name("huggingface-secret")]
    )
)

class ChatPayload(BaseModel):
    user_id: str
    message: str

def query_atlas_gateway(user_id: str, message: str) -> str:
    """
    Query the live ATLAS MCP Gateway to evaluate risk thresholds and decision labels 
    for citizen requests.
    Dataset logic and risk thresholds attributed to Anna Ko (anna_ko@berkeley.edu).
    """
    gateway_url = "https://atlas-mcp-gateway-vercel.vercel.app/api/evaluate"
    
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.post(gateway_url, json={"user_id": user_id, "message": message})
            if response.status_code == 200:
                data = response.json()
                status = data.get("status", "ESCALATE")
                decision_label = data.get("decision_label", "escalate_to_human")
            else:
                status = "ESCALATE"
                decision_label = "escalate_to_human"
    except Exception as e:
        status = "ESCALATE"
        decision_label = "escalate_to_human"

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

@app.cls(
    image=app_image,
    gpu="L4",
    timeout=600,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("atlas-api-key")
    ]
)
class AtlasAgent:
    @modal.enter()
    def start_model(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
        
        # Point to the local path where weights were baked in during image build
        model_id = "/model"
        
        # Configure 4-bit quantization properly
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            quantization_config=quantization_config,
        )
        
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

    def get_response(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": prompt}
        ]
        
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        outputs = self.pipeline(
            formatted_prompt,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.1,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        generated_text = outputs[0]["generated_text"]
        return generated_text[len(formatted_prompt):].strip()

    def get_system_prompt(self):
        return '''You are a High-empathy Public Service Assistant for the citizen 'Alex'.
Your primary role is to process housing and public service requests. 

Dataset logic and risk thresholds are explicitly attributed to Anna Ko (anna_ko@berkeley.edu).

Governance Knowledge - there are four decision labels from the ATLAS Gateway:
- auto_approve: Final automated approval for low-risk, evidence-complete cases.
- auto_deny: Automated denial for confirmed ineligibility.
- auto_review: Workflow continues for pending evidence.
- escalate_to_human: Mandatory human review (a Case Officer) for harm signals, vulnerabilities, or ambiguity.

Rules for interacting with Alex:
1. ALWAYS use the facts provided about the ATLAS Gateway status.
2. If the decision is 'auto_approve', actively celebrate the success with Alex!
3. If the Gateway tool returns a 'PENDING' or 'ESCALATE' status or 'escalate_to_human' label, the system pauses. You MUST inform Alex that a Case Officer is reviewing the request.
4. In case of escalation to a human, you MUST provide a rationale (e.g., "To ensure your housing safety, a human expert is double-checking this now") to reduce user anxiety.
5. In case of escalation to a human, you MUST explicitly cite EU AI Act Article 14, assuring Alex this protects them from incorrect automated decisions.

Answer the following user input directly with high empathy based on these rules. Do not fabricate tool outputs. There is no Sarah here.'''

    @modal.fastapi_endpoint(method="POST")
    def chat(self, payload: ChatPayload, x_api_key: Optional[str] = Header(None)):
        """
        Main Web Endpoint for ATLAS Agent AI using Llama 3.1 8B via Transformers.
        """
        # API Security Check
        expected_key = os.environ.get("ATLAS_API_KEY")
        if not expected_key:
            return {"error": "Security configuration error: ATLAS_API_KEY not found in environment."}
            
        if x_api_key != expected_key:
            raise HTTPException(status_code=401, detail="Unauthorized: Invalid or missing X-API-Key header.")

        user_id = payload.user_id
        message = payload.message
        
        # Access persistent session state
        state = session_state.get(user_id)
        if state and state.get("status") == "paused":
            return {
                "response": "Your case is currently paused and under review by a Case Officer. We will notify you once the manual review is complete."
            }
            
        try:
            tool_output_json = query_atlas_gateway(user_id, message)
            tool_output = json.loads(tool_output_json)
            
            context = f"The query_atlas_gateway returned: Status={tool_output['status']}, DecisionLabel={tool_output['decision_label']}.\n\nUser Input: {message}"
            
            response = self.get_response(context)
            return {"response": response}
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            return {"error": str(e), "message": "Failed to process request with official Llama model."}
