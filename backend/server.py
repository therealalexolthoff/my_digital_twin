from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os
from dotenv import load_dotenv
from typing import Optional, List, Dict
import json
import uuid
from datetime import datetime
from pathlib import Path

load_dotenv(override=True)

app = FastAPI()

origins = os.getenv("CORS_ORIGINS","http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

client=OpenAI()

MEMORY_DIR= Path("../memory")
MEMORY_DIR.mkdir(exist_ok=True)


def load_personality():
    with open("alex_sales_bot.txt", "r", encoding="utf-8") as f:
        return f.read().strip()

PERSONALITY=load_personality()

def load_conversation(session_id: str) ->List[Dict]:
    """Load conversation history from file"""
    file_path = MEMORY_DIR / f"{session_id}.json"
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_conversation(session_id: str, messages: list([Dict])):
    """Save conversation history to file"""
    file_path = MEMORY_DIR / f"{session_id}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(messages, f, indent=2, ensure_ascii=False)
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str

@app.get("/")
async def root():
    return {"message": "AI Digital Twin API with Memory"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        session_id = request.session_id or str(uuid.uuid4())

        conversation=load_conversation(session_id)

        messages = [
            {"role": "system", "content": PERSONALITY},
        ]

        for msg in conversation:
            messages.append(msg)

        messages.append({"role": "user", "content": request.message})

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )

        assistant_response = response.choices[0].message.content

        conversation.append({"role": "user", "content": request.message})
        conversation.append({"role": "assistant", "content": assistant_response})

        save_conversation(session_id, conversation)

        return ChatResponse(
            response=assistant_response,
            session_id=session_id
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions")
async def list_sessions():
    """List all conversation sessions"""
    sessions = []
    for file_path in MEMORY_DIR.glob("*.json"):
        session_id = file_path.stem
        with open(file_path, "r", encoding="utf-8") as f:
            conversation = json.load(f)
            sessions.append({
                "session_id": session_id,
                "message_count": len(conversation),
                "last_message": conversation[-1]["content"] if conversation else None
            })
    return {"sessions": sessions}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)