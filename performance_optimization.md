# Оптимизация производительности

## Текущие узкие места

1. **Загрузка моделей**:
   - Модели Whisper требуют значительного времени для загрузки
   - Модели диаризации (PyAnnote) также тяжеловесны

2. **Обработка аудио**:
   - Обработка длинных аудиофайлов занимает много времени
   - Высокие требования к памяти

3. **Диаризация**:
   - Алгоритмы диаризации требуют значительных вычислительных ресурсов
   - Качество диаризации может быть неудовлетворительным при определенных условиях

## Стратегии оптимизации

### 1. Оптимизация моделей ASR (распознавания речи)

#### Квантизация моделей
```python
# Пример использования более легких квантованных моделей
from faster_whisper import WhisperModel

model = WhisperModel("small", device="cpu", compute_type="int8")
```

#### Разбиение на чанки
```python
# Текущий подход
segments, info = model.transcribe(
    audio_path,
    language=lang,
    vad_filter=False,
    beam_size=1,
    chunk_length=15,  # Можно оптимизировать
    word_timestamps=False
)

# Оптимизированный подход с параллельной обработкой
def process_chunk(chunk_data, model, lang):
    return model.transcribe(chunk_data, language=lang, beam_size=1)

from concurrent.futures import ProcessPoolExecutor
with ProcessPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(process_chunk, chunk, model, lang) 
               for chunk in audio_chunks]
    results = [future.result() for future in futures]
```

### 2. Оптимизация диаризации

#### Использование более легких моделей
```python
# Вместо тяжелой PyAnnote можно использовать облегченные альтернативы
# для предварительной диаризации, а затем уточнять результаты
```

#### Предварительная обработка аудио
```python
# Удаление тишины перед диаризацией
import librosa
import numpy as np

def remove_silence(audio, sr, threshold_db=-40):
    intervals = librosa.effects.split(audio, top_db=-threshold_db)
    processed_audio = np.concatenate([audio[start:end] for start, end in intervals])
    return processed_audio
```

### 3. Параллельная обработка

#### Разбиение длинных файлов
```python
def split_audio(audio_path, chunk_minutes=5):
    """Разбивает аудиофайл на чанки заданной длительности"""
    import subprocess
    import os
    from pathlib import Path
    
    output_dir = Path("audio_chunks")
    output_dir.mkdir(exist_ok=True)
    
    # Получение длительности
    cmd = f"ffprobe -i {audio_path} -show_entries format=duration -v quiet -of csv='p=0'"
    duration = float(subprocess.check_output(cmd, shell=True).decode().strip())
    
    chunk_files = []
    chunk_sec = chunk_minutes * 60
    
    for i in range(0, math.ceil(duration / chunk_sec)):
        start = i * chunk_sec
        if start >= duration:
            break
            
        output_file = output_dir / f"chunk_{i:03d}.wav"
        cmd = f"ffmpeg -hide_banner -y -ss {start} -t {chunk_sec} -i {audio_path} -c:a pcm_s16le -ar 16000 -ac 1 {output_file}"
        subprocess.run(cmd, shell=True)
        chunk_files.append(output_file)
    
    return chunk_files
```

#### Параллельная обработка чанков
```python
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

def process_audio_file(file_path):
    # Обработка одного файла
    # ...
    return results

# Определение количества процессов
num_processes = max(1, multiprocessing.cpu_count() - 1)

with ProcessPoolExecutor(max_workers=num_processes) as executor:
    audio_chunks = split_audio(input_audio, chunk_minutes=5)
    futures = [executor.submit(process_audio_file, chunk) for chunk in audio_chunks]
    results = [future.result() for future in futures]

# Объединение результатов
combined_results = combine_results(results)
```

### 4. Кэширование промежуточных результатов

```python
import os
import json
from pathlib import Path

def get_cached_result(audio_file, step_name):
    """Получение кэшированных результатов, если они существуют"""
    cache_file = Path(f"cache/{os.path.basename(audio_file)}_{step_name}.json")
    if cache_file.exists():
        with open(cache_file, "r") as f:
            return json.load(f)
    return None

def save_cache(audio_file, step_name, data):
    """Сохранение результатов в кэш"""
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f"{os.path.basename(audio_file)}_{step_name}.json"
    with open(cache_file, "w") as f:
        json.dump(data, f)
```

## Рекомендации по реализации

1. **Поэтапный подход**:
   - Внедрять оптимизации постепенно, начиная с наиболее узких мест
   - Измерять результаты каждой оптимизации

2. **Мониторинг ресурсов**:
   - Отслеживать использование памяти и CPU/GPU
   - Адаптировать параметры в зависимости от доступных ресурсов

3. **Гибкая настройка**:
   - Предоставить пользователю возможность выбора между качеством и скоростью
   - Автоматический выбор параметров на основе характеристик входного файла

4. **Баланс между скоростью и качеством**:
   - Более быстрые модели могут давать менее точные результаты
   - Предложить несколько профилей обработки (быстрый/сбалансированный/качественный)
