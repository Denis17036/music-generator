document.addEventListener('DOMContentLoaded', () => {
  const generateBtn = document.getElementById('generate');
  const promptInput = document.getElementById('prompt');
  const durationInput = document.getElementById('duration');
  const styleSelect = document.getElementById('style');
  const audioPlayer = document.getElementById('player');
  const downloadLink = document.createElement('a');
  
  downloadLink.textContent = 'Скачать';
  downloadLink.style.display = 'none';
  document.querySelector('.container').appendChild(downloadLink);

  generateBtn.addEventListener('click', async () => {
    const prompt = promptInput.value.trim();
    if (!prompt) {
      alert('Введите описание музыки!');
      return;
    }

    generateBtn.disabled = true;
    generateBtn.textContent = 'Генерация...';
    
    try {
      const response = await fetch('/api/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          prompt: prompt,
          duration: parseFloat(durationInput.value),
          style: styleSelect.value
        })
      });

      if (!response.ok) throw new Error('Ошибка генерации');
      
      const data = await response.json();
      const audioBlob = new Blob([Uint8Array.from(atob(data.audio), c => c.charCodeAt(0))], { type: 'audio/mp3' });
      
      audioPlayer.src = URL.createObjectURL(audioBlob);
      downloadLink.href = audioPlayer.src;
      downloadLink.download = 'generated_music.mp3';
      downloadLink.style.display = 'inline-block';
      
    } catch (error) {
      console.error('Error:', error);
      alert('Произошла ошибка: ' + error.message);
    } finally {
      generateBtn.disabled = false;
      generateBtn.textContent = 'Сгенерировать';
    }
  });
});
