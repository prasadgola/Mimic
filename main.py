import os
import json
import asyncio
import base64
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

# --- System Prompt (Your Digital Twin Profile) ---
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

# --- Gemini Client ---
def get_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    return genai.Client(api_key=api_key)


# --- Text Chat Endpoint ---
class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []  # [{"role": "user"|"model", "text": "..."}]


class ChatResponse(BaseModel):
    response: str


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """HTTP endpoint for text-based conversation with the digital twin."""
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
        model="gemini-2.5-flash-preview-05-20",
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.7,
            max_output_tokens=1024,
        ),
    )

    return ChatResponse(response=response.text)


# --- Voice WebSocket Endpoint (Gemini Live API) ---
@app.websocket("/voice")
async def voice_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time voice conversation.

    Protocol:
    - Client sends: binary audio chunks (PCM 16-bit, 16kHz, mono)
    - Server sends: binary audio chunks (PCM 16-bit, 24kHz, mono)
    - Server sends: JSON {"type": "transcript", "text": "..."}  ← AI transcript chunks
    - Server sends: JSON {"type": "turn_complete"}              ← end of AI turn
    - Server sends: JSON {"type": "error", "message": "..."}   ← on failure
    - Client sends: JSON {"type": "close"}                     ← to end session
    """
    await websocket.accept()
    client = get_client()

    try:
        # FIX 1: Request both AUDIO and TEXT so response.text is populated
        config = types.LiveConnectConfig(
            response_modalities=["AUDIO", "TEXT"],
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
        )

        async with client.aio.live.connect(
            model="gemini-2.5-flash-native-audio-preview-12-2025",
            config=config,
        ) as session:

            async def receive_and_forward_audio():
                """Receive audio/control messages from Android and forward to Gemini."""
                try:
                    while True:
                        data = await websocket.receive()

                        if "bytes" in data:
                            await session.send(
                                input=types.LiveClientRealtimeInput(
                                    media_chunks=[
                                        types.Blob(
                                            data=data["bytes"],
                                            mime_type="audio/pcm;rate=16000",
                                        )
                                    ]
                                )
                            )

                        elif "text" in data:
                            msg = json.loads(data["text"])
                            if msg.get("type") == "close":
                                await session.close()
                                return
                            # "end" type is a no-op for now (Gemini VAD handles turn detection)

                except WebSocketDisconnect:
                    await session.close()

            async def receive_and_send_response():
                """Receive from Gemini and forward audio + transcript to Android."""
                try:
                    async for response in session.receive():

                        # ── DEBUG: log the full response shape to Cloud Run logs ──
                        print(
                            f"[Gemini] data={bool(response.data)} "
                            f"text={repr(response.text)} "
                            f"server_content={response.server_content}"
                        )

                        # ── 1. Audio bytes → send as binary to Android ────────────
                        if response.data:
                            await websocket.send_bytes(response.data)

                        # ── 2a. Top-level response.text (SDK convenience field) ───
                        if response.text:
                            await websocket.send_text(
                                json.dumps({"type": "transcript", "text": response.text})
                            )

                        # ── 2b. Nested inside server_content.model_turn.parts ─────
                        #   Some SDK versions put transcript here instead of top-level
                        if response.server_content and response.server_content.model_turn:
                            for part in (response.server_content.model_turn.parts or []):
                                if hasattr(part, "text") and part.text:
                                    # Avoid double-sending if top-level already sent it
                                    if not response.text:
                                        await websocket.send_text(
                                            json.dumps({"type": "transcript", "text": part.text})
                                        )

                        # ── 3. Turn complete signal → Android reconnects ──────────
                        if response.server_content and response.server_content.turn_complete:
                            await websocket.send_text(
                                json.dumps({"type": "turn_complete"})
                            )

                except Exception as e:
                    print(f"[Gemini] Error in receive_and_send_response: {e}")

            await asyncio.gather(
                receive_and_forward_audio(),
                receive_and_send_response(),
            )

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Voice session error: {e}")
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))
        except Exception:
            pass


# --- Health Check ---
@app.get("/health")
async def health():
    return {"status": "ok", "service": "digital-twin"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)