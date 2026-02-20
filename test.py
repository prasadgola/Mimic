import asyncio
import websockets
import sounddevice as sd
import numpy as np

async def test():
    uri = "wss://basavaprasad-digital-twin-882178443942.us-central1.run.app/voice"
    
    async with websockets.connect(uri) as ws:
        print("Connected! Speak now... (5 seconds)")
        
        # Record 5 seconds of audio (16kHz, mono, 16-bit)
        duration = 5
        audio = sd.rec(int(duration * 16000), samplerate=16000, channels=1, dtype='int16')
        
        # Send chunks while recording
        for i in range(50):
            await asyncio.sleep(0.1)
            chunk_start = i * 1600
            chunk_end = chunk_start + 1600
            if chunk_end <= len(audio):
                await ws.send(audio[chunk_start:chunk_end].tobytes())
        
        sd.wait()
        print("Done speaking, sending end signal...")
        await ws.send('{"type": "end"}')
        
        print("Waiting for response...")
        try:
            while True:
                response = await asyncio.wait_for(ws.recv(), timeout=15)
                if isinstance(response, bytes):
                    print(f"Got audio: {len(response)} bytes")
                else:
                    print(f"Got text: {response}")
                    if "turn_complete" in response:
                        break
        except asyncio.TimeoutError:
            print("Timeout - no response")

asyncio.run(test())