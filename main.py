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

    # Build conversation history
    contents = []
    for msg in request.history:
        contents.append(
            types.Content(
                role=msg["role"],
                parts=[types.Part.from_text(text=msg["text"])],
            )
        )
    # Add current message
    contents.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=request.message)],
        )
    )

    response = await asyncio.to_thread(
        client.models.generate_content,
        model="gemini-3-flash-preview",
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
    - Server sends: binary audio chunks back (PCM 16-bit, 24kHz, mono)
    - Client sends: JSON {"type": "end"} to signal end of turn
    - Client sends: JSON {"type": "close"} to close connection
    """
    await websocket.accept()

    client = get_client()

    try:
        # Create a live session with Gemini
        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            system_instruction=types.Content(
                parts=[types.Part.from_text(text=SYSTEM_PROMPT)]
            ),
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name="Kore"  # A natural-sounding voice
                    )
                )
            ),
        )

        async with client.aio.live.connect(
            model="gemini-3-flash-preview",
            config=config,
        ) as session:

            async def receive_and_forward_audio():
                """Receive audio from client and send to Gemini."""
                try:
                    while True:
                        data = await websocket.receive()

                        if "bytes" in data:
                            # Binary audio data from client
                            audio_bytes = data["bytes"]
                            await session.send(
                                input=types.LiveClientRealtimeInput(
                                    media_chunks=[
                                        types.Blob(
                                            data=audio_bytes,
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
                            elif msg.get("type") == "end":
                                # End of user's turn
                                pass

                except WebSocketDisconnect:
                    await session.close()

            async def receive_and_send_response():
                """Receive audio from Gemini and send to client."""
                try:
                    async for response in session.receive():
                        if response.data:
                            # Send raw audio bytes back to client
                            await websocket.send_bytes(response.data)

                        if response.server_content and response.server_content.turn_complete:
                            # Signal turn complete to client
                            await websocket.send_text(
                                json.dumps({"type": "turn_complete"})
                            )

                except Exception as e:
                    print(f"Error receiving from Gemini: {e}")

            # Run both tasks concurrently
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
        except:
            pass


# --- Health Check ---
@app.get("/health")
async def health():
    return {"status": "ok", "service": "digital-twin"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
