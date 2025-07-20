import torch # type: ignore
from transformers import AutoProcessor, MusicgenForConditionalGeneration # type: ignore
from pydub import AudioSegment # type: ignore
import io

def generate_music(prompt: str, duration: float = 60.0, style: str = "electronic"):
    # Загрузка модели (кешируется после первого раза)
    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    model = MusicgenForConditionalGeneration.from_pretrained(
        "facebook/musicgen-small",
        device_map="cpu",
        torch_dtype=torch.float32
    )
    
    # Генерация по частям
    full_audio = []
    for _ in range(int(duration / 10)):  # 10-секундные отрезки
        inputs = processor(
            text=[f"{prompt}, {style} style"],
            padding=True,
            return_tensors="pt",
        )
        audio = model.generate(**inputs, max_new_tokens=256)
        full_audio.append(audio[0, 0].numpy())
    
    # Сборка в MP3
    audio_segment = AudioSegment.from_wav(io.BytesIO(b''))  # Пустышка
    for chunk in full_audio:
        wav_io = io.BytesIO()
        wav.write(wav_io, 32000, chunk) # type: ignore
        audio_segment += AudioSegment.from_wav(wav_io)
    
    mp3_io = io.BytesIO()
    audio_segment.export(mp3_io, format="mp3")
    return mp3_io.getvalue()
