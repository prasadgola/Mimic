# Digital Twin Server

FastAPI backend that serves as Basavaprasad's digital twin — responds to text and voice as him using Gemini.

## Endpoints

| Endpoint | Protocol | Purpose |
|----------|----------|---------|
| `POST /chat` | HTTP | Text conversation |
| `WS /voice` | WebSocket | Real-time voice (audio in/out) |
| `GET /health` | HTTP | Health check |

## Text Chat (`/chat`)

```json
// Request
POST /chat
{
    "message": "Tell me about your projects",
    "history": [
        {"role": "user", "text": "Hi"},
        {"role": "model", "text": "Hey! What's up?"}
    ]
}

// Response
{
    "response": "I've been working on a few things..."
}
```

## Voice (`/voice`)

WebSocket protocol:
- **Client → Server**: Binary PCM audio (16-bit, 16kHz, mono)
- **Server → Client**: Binary PCM audio (16-bit, 24kHz, mono)
- **Client → Server**: `{"type": "close"}` to end session
- **Server → Client**: `{"type": "turn_complete"}` when model finishes speaking

## Local Development

```bash
export GEMINI_API_KEY="your-key"
pip install -r requirements.txt
uvicorn main:app --reload --port 8080
```

## Deploy to Cloud Run

1. Edit `deploy.sh` — set your `PROJECT_ID` and `GEMINI_API_KEY`
2. Run: `chmod +x deploy.sh && ./deploy.sh`

## Android Integration

- Text: Use Retrofit/OkHttp to call `POST /chat`
- Voice: Use OkHttp WebSocket to connect to `/voice`, stream PCM audio from `AudioRecord`, play response with `AudioTrack`

## Important Notes

- Cloud Run supports WebSockets with `--session-affinity` flag
- Voice uses Gemini Live API (`gemini-2.0-flash-live-001`) for real-time audio
- The system prompt in `main.py` defines the digital twin persona — customize as needed
