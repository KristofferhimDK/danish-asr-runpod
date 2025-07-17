import runpod
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import base64
import tempfile
import os

print("✅ handler.py started")

model_id = "syvai/hviske-v2"
device = "cpu"
torch_dtype = torch.float32

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
    device=-1
)

def handler(job):
    audio_base64 = job["input"]["audio"]
    audio_bytes = base64.b64decode(audio_base64)
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    
    result = pipe(tmp_path)
    os.unlink(tmp_path)
    
    return {"transcription": result["text"]}

runpod.serverless.start({"handler": handler})
