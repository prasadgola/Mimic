import asyncio
import websockets

async def test_ws():
    uri = "wss://basavaprasad-digital-twin-882178443942.us-central1.run.app/voice"
    print(f"Connecting to {uri}...")
    
    try:
        async with websockets.connect(uri) as ws:
            print("✅ Connected to Cloud Run!")
            
            # Send 1 second of silent 16kHz PCM audio
            print("Sending 1 second of dummy audio to Gemini...")
            dummy_audio = b'\x00' * 32000 
            await ws.send(dummy_audio)
            
            print("Waiting for response...")
            while True:
                response = await ws.recv()
                if isinstance(response, bytes):
                    print(f"✅ Success! Received {len(response)} bytes of audio from Gemini.")
                    break
                else:
                    print(f"Received text payload: {response}")
                    
    except websockets.exceptions.InvalidStatusCode as e:
        print(f"❌ Handshake failed (Status {e.status_code}). Check Cloud Run permissions/auth.")
    except Exception as e:
        print(f"❌ Error: {e}")

asyncio.run(test_ws())