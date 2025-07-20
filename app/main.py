import os
import sys
import logging
from fastapi import FastAPI
from pydantic import BaseModel
import subprocess

# Настройка логгера
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Проверка и установка зависимостей при необходимости
def install_dependencies():
    required_packages = {
        "uvicorn": "uvicorn[standard]==0.27.0",
        "fastapi": "fastapi==0.109.2",
        "torch": "torch==2.0.1+cpu",
        "transformers": "transformers==4.30.2",
        "scipy": "scipy==1.10.1",
        "pydub": "pydub==0.25.1"
    }
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✅ {package} установлен")
        except ImportError:
            missing_packages.append(required_packages[package])
            logger.warning(f"❌ {package} не найден")
    
    if missing_packages:
        logger.info("Устанавливаем отсутствующие зависимости...")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install"] + missing_packages,
                check=True
            )
            logger.info("✅ Зависимости успешно установлены")
            # Перезапуск приложения после установки
            os.execv(sys.executable, [sys.executable] + sys.argv)
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Ошибка установки зависимостей: {e}")
            sys.exit(1)

# Вызываем проверку зависимостей перед созданием приложения
install_dependencies()

# Импортируем остальные зависимости после проверки
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import torch
import scipy.io.wavfile as wav
from pydub import AudioSegment
import io
import numpy as np

# Создаем FastAPI приложение
app = FastAPI(title="🎵 Генератор музыки", version="1.0")

# Загрузка модели (выполняется при первом запросе)
model = None
processor = None

class GenerationRequest(BaseModel):
    prompt: str
    duration: float = 30.0  # Длина в секундах
    temperature: float = 1.0
    guidance_scale: float = 3.0

def load_model():
    global model, processor
    if model is None:
        logger.info("⏳ Загрузка модели...")
        model = MusicgenForConditionalGeneration.from_pretrained(
            "facebook/musicgen-small",
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
        logger.info("✅ Модель загружена")

@app.get("/")
async def home():
    return {
        "message": "Добро пожаловать в генератор музыки!",
        "endpoints": {
            "generate": "POST /generate",
            "health": "GET /health"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "ok", "torch_available": torch.cuda.is_available()}

@app.post("/generate")
async def generate_music(request: GenerationRequest):
    try:
        load_model()  # Гарантируем загрузку модели
        
        logger.info(f"🔮 Генерация музыки: {request.prompt[:50]}...")
        logger.info(f"Параметры: duration={request.duration}s, temp={request.temperature}")
        
        # Рассчитываем количество чанков
        num_chunks = max(1, int(request.duration / 10))
        
        full_audio = []
        for i in range(num_chunks):
            inputs = processor(
                text=[request.prompt],
                padding=True,
                return_tensors="pt",
            )
            
            audio_chunk = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=request.temperature,
                guidance_scale=request.guidance_scale,
            )
            
            full_audio.append(audio_chunk[0, 0].numpy())
            logger.info(f"✅ Сгенерирован чанк {i+1}/{num_chunks}")
        
        # Собираем в один трек
        audio_array = np.concatenate(full_audio)
        sampling_rate = model.config.audio_encoder.sampling_rate
        
        # Конвертация в MP3
        wav_io = io.BytesIO()
        wav.write(wav_io, sampling_rate, audio_array)
        wav_io.seek(0)
        
        mp3_io = io.BytesIO()
        AudioSegment.from_wav(wav_io).export(mp3_io, format="mp3")
        mp3_io.seek(0)
        
        logger.info("🎵 Генерация завершена")
        return {"audio": mp3_io.getvalue().hex()}  # Возвращаем hex для JSON
        
    except Exception as e:
        logger.error(f"❌ Ошибка генерации: {str(e)}")
        return {"error": str(e)}

# Запуск через Uvicorn (если файл запущен напрямую)
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
