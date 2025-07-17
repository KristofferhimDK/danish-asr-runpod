import runpod
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import base64
import tempfile
import os

print("📦 Starting model setup...")

model_id = "syvai/hviske-v2"
device = "cpu"  # Force CPU for debugging
torch_dtype = torch.float32

print("📥 Loading model...")
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    use_auth_token=os.environ.get("HF_TOKEN")
)

print("🚀 Moving model to device...")
model.to(device)

print("🔧 Loading processor...")
processor = AutoProcessor.from_pretrained(model_id)

print("🛠️ Setting up pipeline...")
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=-1  # Use CPU
)

print("✅ Setup complete.")

def handler(job):
    print("📩 Job received.")
    try:
        audio_base64 = job["input"]["audio"]
        audio_bytes = base64.b64decode(audio_base64)
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        
        print("🔊 Running ASR...")
        result = pipe(tmp_path)
        os.unlink(tmp_path)
        
        print("📝 Transcription done.")
        return {"transcription": result["text"]}
    except Exception as e:
        print("❌ Exception:", str(e))
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
