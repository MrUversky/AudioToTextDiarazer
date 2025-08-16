#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Инструмент для разбиения длинных аудиофайлов на более короткие части
для более эффективной обработки в AudioToTextDiarazer
"""

import os
import sys
import math
import argparse
import subprocess
from pathlib import Path


def to_hhmmss(seconds: float) -> str:
    """Преобразует секунды в формат HH:MM:SS"""
    s = max(0.0, float(seconds))
    h = int(s // 3600); s -= 3600*h
    m = int(s // 60);   s -= 60*m
    return f"{h:02d}:{m:02d}:{int(round(s)):02d}"


def get_duration(input_file: str) -> float:
    """Получает длительность аудио/видео файла в секундах"""
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", 
           "-of", "default=noprint_wrappers=1:nokey=1", input_file]
    try:
        output = subprocess.check_output(cmd).decode().strip()
        return float(output)
    except subprocess.CalledProcessError as e:
        print(f"Ошибка при получении длительности файла: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Не удалось преобразовать длительность в число: {e}")
        sys.exit(1)


def split_audio(input_file: str, output_dir: str, chunk_minutes: int = 15):
    """Разбивает длинный аудиофайл на более короткие части заданной длительности"""
    input_path = Path(input_file)
    
    if not input_path.exists():
        print(f"Файл не найден: {input_file}")
        sys.exit(1)
    
    # Создаем директорию для фрагментов
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Получаем длительность файла
    duration_sec = get_duration(input_file)
    total_chunks = math.ceil(duration_sec / (chunk_minutes * 60))
    
    print(f"Файл: {input_file}")
    print(f"Длительность: {to_hhmmss(duration_sec)} ({duration_sec:.2f} сек)")
    print(f"Размер фрагмента: {chunk_minutes} минут")
    print(f"Количество фрагментов: {total_chunks}")
    
    chunk_files = []
    chunk_sec = chunk_minutes * 60
    
    for i in range(total_chunks):
        start_sec = i * chunk_sec
        if start_sec >= duration_sec:
            break
            
        # Формируем имя выходного файла
        output_file = output_path / f"chunk_{i+1:03d}_{input_path.stem}.wav"
        
        # Формируем команду для ffmpeg
        cmd = [
            "ffmpeg", "-hide_banner", "-y",
            "-ss", str(start_sec),
            "-t", str(chunk_sec),
            "-i", str(input_path),
            "-vn",  # Без видео
            "-ac", "1",  # Моно
            "-ar", "16000",  # Частота дискретизации 16кГц
            "-c:a", "pcm_s16le",  # Формат аудио
            str(output_file)
        ]
        
        print(f"\nОбработка фрагмента {i+1}/{total_chunks}...")
        print(f"  Начало: {to_hhmmss(start_sec)}")
        print(f"  Выходной файл: {output_file}")
        
        try:
            subprocess.run(cmd, check=True)
            chunk_files.append(output_file)
            print(f"  ✓ Фрагмент {i+1} готов")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Ошибка при обработке фрагмента {i+1}: {e}")
    
    print(f"\nГотово! Создано {len(chunk_files)} фрагментов в директории {output_path}")
    return chunk_files


def main():
    parser = argparse.ArgumentParser(description="Разбиение аудио/видео на фрагменты")
    parser.add_argument("input", help="Путь к входному аудио/видео файлу")
    parser.add_argument("-o", "--output-dir", default="audio_chunks",
                        help="Директория для сохранения фрагментов (по умолчанию: audio_chunks)")
    parser.add_argument("-m", "--minutes", type=int, default=15,
                        help="Длительность каждого фрагмента в минутах (по умолчанию: 15)")
    
    args = parser.parse_args()
    
    # Проверка наличия ffmpeg
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        print("Ошибка: ffmpeg не установлен. Установите ffmpeg для работы этого скрипта.")
        sys.exit(1)
    
    split_audio(args.input, args.output_dir, args.minutes)


if __name__ == "__main__":
    main()
