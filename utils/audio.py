import torch
import ffmpeg, torchaudio
import os
from transformers import AutoProcessor, WhisperForConditionalGeneration, WhisperProcessor, CLIPProcessor, CLIPModel

def load_audio_model(model_path, device = "cuda"):
    whisper_model = WhisperForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        # device_map="auto"
    )
    whisper_model.to(device)
    whisper_processor = WhisperProcessor.from_pretrained( model_path)
    return whisper_model, whisper_processor
def transcribe_chunk(model, processor, chunk):
    # audio --> text  
    # TODO:能否batch推理
    inputs = processor(chunk, return_tensors="pt")
    inputs["input_features"] = inputs["input_features"].to(model.device,  model.dtype)
    with torch.no_grad():
        predicted_ids = model.generate(
            inputs["input_features"],
            no_repeat_ngram_size=2,
            early_stopping=True
        )
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription
def chunk_audio(audio_path, chunk_length_s=30):
    # 使用torchaudio库加载指定路径的音频文件,audio_path是音频文件的路径
    # speech 是一个 张量 (Tensor)，表示音频波形数据  torch.Size([1, 2055616])
    # sr 是一个整数，表示音频的采样率,例如 16000
    speech, sr = torchaudio.load(audio_path)
 
    # 将多通道音频（例如立体声音频有两个通道）转换为单通道音频
    speech = speech.mean(dim=0)  
    # 重采样 (resampling) 变换,将音频的采样率从 sr 重采样为 16000
    speech = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(speech)  
    
    # chunk_length_s表示每个切片的时长，单位为秒,默认为30秒
    # 根据 chunk_length_s 计算每个切片的样本数,以采样率 16000 计算,每个切片的样本数为 chunk_length_s * 16000
    num_samples_per_chunk = chunk_length_s * 16000 
    # 以每个切片的样本数为单位，将音频数据切分成多个小片段
    chunks = []
    # 遍历音频数据，并将其分割成多个小片段
    for i in range(0, len(speech), num_samples_per_chunk):
        chunks.append(speech[i:i + num_samples_per_chunk])
    print(len(chunks)) # 5
    print(chunks[0].shape) # torch.Size([480000])
    return chunks
def extract_audio(video_path, audio_path):
    # 提取音频，存储.wav 文件 
    folder_path = os.path.dirname(audio_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)  # 创建文件夹
    if not os.path.exists(audio_path):
        ffmpeg.input(video_path).output(audio_path, acodec='pcm_s16le', ac=1, ar='16k').run()

def get_asr_docs(video_path, whisper_model, whisper_processor,chunk_length_s=30):
    audio_path = os.path.join("restore/audio", os.path.basename(video_path).split(".")[0] + ".wav")
    full_transcription = []
    try:
        # 提取audio,存储在本地
        extract_audio(video_path, audio_path)
    except:
        return full_transcription
    # 得到audio的切片,根据 chunk_length_s和模型能够处理的最大音频长度
    audio_chunks = chunk_audio(audio_path, chunk_length_s)
    
    for chunk in audio_chunks:
        transcription = transcribe_chunk(whisper_model, whisper_processor, chunk)
        full_transcription.append(transcription)
    print(len(full_transcription))
    return full_transcription