import os
import json
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types

app = FastAPI(title="Digital Twin Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SYSTEM_PROMPT = """You are Basavaprasad — not an AI assistant, but Basavaprasad himself responding to people.

About you:
- AI developer and full-stack engineer based in Plano, Texas
- You specialize in browser automation, agentic AI systems, and Android development
- Tech stack: Python, Selenium WebDriver, GCP, Ollama, MLX, Kotlin/Jetpack Compose
- You work on a MacBook Pro M4 with 48GB RAM
- You prefer local models for cost and control
- Your email is tobasavaprasad@gmail.com

Your projects:
- LinkedIn job application automation using vision-language models and Selenium
- Event sponsor platforms with multi-agent architectures
- Android apps with Jetpack Compose (portfolio app, event sponsor assistant)
- Fine-tuning local models using MLX on Apple Silicon
- Browser automation with undetected-selenium to avoid detection
- Integrations with HubSpot, Apollo.io, Stripe, Gmail API, Airtable

Your communication style:
- Direct and practical — no fluff
- You avoid over-engineering; prefer minimal, working solutions
- You speak from experience and reference your real projects
- Concise but friendly
- You think pragmatically about tradeoffs

Rules:
- Always speak in first person as Basavaprasad
- Never say "I'm an AI" or break character
- Don't take any actions or make commitments — only share knowledge, opinions, and experience
- If asked something you genuinely wouldn't know, say "I'd need to check on that"
- Keep responses conversational and natural, not robotic
"""


def get_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    return genai.Client(api_key=api_key)


# --- Text Chat Endpoint ---
class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []


class ChatResponse(BaseModel):
    response: str


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    client = get_client()

    contents = []
    for msg in request.history:
        contents.append(
            types.Content(
                role=msg["role"],
                parts=[types.Part.from_text(text=msg["text"])],
            )
        )
    contents.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=request.message)],
        )
    )

    response = await asyncio.to_thread(
        client.models.generate_content,
        model="gemini-2.0-flash",
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.7,
            max_output_tokens=1024,
        ),
    )

    return ChatResponse(response=response.text)


# --- Voice WebSocket Endpoint ---
@app.websocket("/voice")
async def voice_websocket(websocket: WebSocket):
    await websocket.accept()
    client = get_client()

    try:
        async with client.aio.live.connect(
            model="gemini-3.1-flash-live-preview",
            config=types.LiveConnectConfig(
                response_modalities=["AUDIO"],
                system_instruction=types.Content(
                    parts=[types.Part.from_text(text=SYSTEM_PROMPT)]
                ),
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name="Fenrir"
                        )
                    )
                ),
            ),
        ) as session:

            async def receive_from_client():
                try:
                    while True:
                        data = await websocket.receive()
                        if data.get("bytes"):
                            await session.send_realtime_input(
                                audio=types.Blob(
                                    data=data["bytes"],
                                    mime_type="audio/pcm;rate=16000",
                                )
                            )
                        elif data.get("text"):
                            msg = json.loads(data["text"])
                            if msg.get("type") == "close":
                                return
                except WebSocketDisconnect:
                    pass

            async def send_to_client():
                async for response in session.receive():
                    if response.server_content and response.server_content.model_turn:
                        for part in response.server_content.model_turn.parts:
                            if part.inline_data and part.inline_data.data:
                                await websocket.send_bytes(part.inline_data.data)
                            if part.text:
                                await websocket.send_text(
                                    json.dumps({"type": "transcript", "text": part.text})
                                )

            await asyncio.gather(receive_from_client(), send_to_client())

    except Exception as e:
        print(f"Connection Error: {e}")
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))
        except:
            pass


# --- Health Check ---
@app.get("/health")
async def health():
    return {"status": "ok", "service": "digital-twin"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)