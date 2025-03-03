from transformers import AutoProcessor, WhisperForConditionalGeneration, WhisperProcessor, CLIPProcessor, CLIPModel
import torch
whisper_model = WhisperForConditionalGeneration.from_pretrained(
        "/root/nfs/codespace/llm-models/MLLM/openai/whisper-large",
        torch_dtype=torch.float16,
        # device_map="auto"
    )