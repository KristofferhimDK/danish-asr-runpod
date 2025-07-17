import runpod
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import base64
import tempfile
import os

print("✅ handler.py started")

# --- Model setup ---
model_id = "syvai/hviske-v2"
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device == "cuda" else torch.float32

print(f"📦 Loading model: {model_id} on {device}")

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    use_auth_token=os.environ.get("HF_TOKEN")
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=0 if device == "cuda" else -1
)

print("✅ Model loaded and pipeline ready")

# --- Request handler ---
def handler(job):
    print("📩 New job received")

    try:
        audio_base64 = job["input"]["audio"]
        audio_bytes = base64.b64decode(audio_base64)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        print("🔊 Running transcription...")
        result = pipe(tmp_path)
        os.unlink(tmp_path)

        print("📝 Done")
        return {"transcription": result["text"]}

    except Exception as e:
        print(f"❌ Error: {e}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
