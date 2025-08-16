# =========================
# ПОЛНЫЙ ПАЙПЛАЙН: ВИДЕО/АУДИО → (ОТ/ДО) → WAV → ASR → (ОПЦ.) ДИАРИЗАЦИЯ
# =========================
import os, sys, time, json, gc, math, subprocess, shlex
from pathlib import Path

# ---------- ПАРАМЕТРЫ ----------
INPUT_MEDIA   = "input.mp4"  # ← твой файл (видео или аудио)
START_MIN     = None                         # например 0.0 или 5.0; None = с начала
END_MIN       = None                         # например 30.0; None = до конца
DIARIZE       = False                         # False → только распознавание (без спикеров)
NUM_SPEAKERS  = 2
LANG          = "ru"

OUT_ROOT      = "run_out"

# ASR-настройки (стабильность/скорость)
PREFER_GPU_ASR        = True                 # если есть CUDA, сначала пробуем GPU
FORCE_CPU_ASR         = True                # True → сразу на CPU (наиболее стабильно)
ASR_MODEL_GPU         = "distil-large-v3"    # "small" ещё безопаснее
ASR_MODEL_CPU         = "small"
ASR_COMPUTE_TYPE_GPU  = "int8_float16"       # безопаснее, чем float16 (меньше памяти)
ASR_CHUNK_LEN_SEC     = 15                   # меньше → стабильнее
ASR_BEAM_SIZE         = 1                    # 1 = быстро, низкая память

# PyANNOTE токен (для лучшей диаризации). 
# Получите токен на https://huggingface.co/pyannote/segmentation-3.0 (кнопка "Access")
# и установите его в окружении перед запуском: export HF_TOKEN="ваш_токен"
# или укажите в блокноте Colab
HF_TOKEN = os.environ.get("HF_TOKEN", "")
# ---------------------------------

# Папки и кэширование
OUT_ROOT = Path(OUT_ROOT); OUT_ROOT.mkdir(parents=True, exist_ok=True)
OUT_ASR  = OUT_ROOT/"asr";  OUT_ASR.mkdir(parents=True, exist_ok=True)
OUT_DIR  = OUT_ROOT/"final";OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = OUT_ROOT/"cache"; CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Функция для проверки наличия кэшированных данных
def get_cached_result(audio_file, step_name):
    """Получение кэшированных результатов, если они существуют"""
    cache_file = CACHE_DIR / f"{Path(audio_file).stem}_{step_name}.json"
    if cache_file.exists():
        print(f"[CACHE] Используются кэшированные данные для {step_name}")
        try:
            with open(cache_file, "r", encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[CACHE] Ошибка чтения кэша: {e}")
    return None

def save_cache(audio_file, step_name, data):
    """Сохранение результатов в кэш"""
    cache_file = CACHE_DIR / f"{Path(audio_file).stem}_{step_name}.json"
    try:
        with open(cache_file, "w", encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[CACHE] Сохранены данные для {step_name}")
    except Exception as e:
        print(f"[CACHE] Ошибка сохранения кэша: {e}")

def pip_install(pkgs):
    cmd = [sys.executable, "-m", "pip", "install", "-q", "-U", *pkgs]
    print("Installing:", " ".join(pkgs))
    subprocess.run(cmd, check=True)

# Установки
try: import faster_whisper  # noqa
except ImportError: pip_install(["faster-whisper"])
try: import whisperx  # noqa
except ImportError: pip_install(["whisperx"])

def to_hhmmss(seconds: float) -> str:
    s = max(0.0, float(seconds))
    h = int(s // 3600); s -= 3600*h
    m = int(s // 60);   s -= 60*m
    return f"{h:02d}:{m:02d}:{int(round(s)):02d}"

def extract_wav(input_media: str, start_min, end_min) -> str:
    try:
        input_path = Path(input_media)
        if not input_path.exists():
            raise FileNotFoundError(f"Нет файла: {input_media}")
            
        out_wav = str(OUT_ROOT / "audio_16k_mono.wav")
        args = ["ffmpeg","-hide_banner","-y","-i", str(input_path)]
        
        if start_min is not None:
            args += ["-ss", to_hhmmss(float(start_min)*60)]
        if end_min is not None:
            if start_min is not None:
                dur = max(0.0, (float(end_min)-float(start_min))*60)
            else:
                dur = max(0.0, float(end_min)*60)
            args += ["-t", f"{dur:.3f}"]
        
        args += ["-vn","-ac","1","-ar","16000","-c:a","pcm_s16le", out_wav]
        print("[FFMPEG]", " ".join(shlex.quote(a) for a in args))
        
        # Выполнение команды с перенаправлением stderr для возможной диагностики
        process = subprocess.run(args, check=False, stderr=subprocess.PIPE, text=True)
        
        if process.returncode != 0:
            error_msg = process.stderr.strip()
            print(f"[FFMPEG ERROR] {error_msg}")
            if "No such file or directory" in error_msg:
                raise FileNotFoundError(f"FFMPEG не может найти файл: {input_media}")
            elif "Invalid data found" in error_msg:
                raise ValueError(f"FFMPEG не может обработать файл: возможно, поврежден или неподдерживаемый формат")
            else:
                raise RuntimeError(f"FFMPEG ошибка (код {process.returncode}): {error_msg[:200]}...")
                
        if not Path(out_wav).exists() or Path(out_wav).stat().st_size == 0:
            raise RuntimeError("WAV файл не был создан или пустой")
            
        return out_wav
    
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        raise
    except ValueError as e:
        print(f"[ERROR] {e}")
        raise
    except Exception as e:
        print(f"[ERROR] Ошибка при извлечении аудио: {e}")
        raise

AUDIO = extract_wav(INPUT_MEDIA, START_MIN, END_MIN)
print("AUDIO:", AUDIO)

# ---------- ASR (faster-whisper) с безопасными настройками ----------
def fmt_ts(t: float) -> str:
    ms = int(round(max(0.0, t) * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def run_asr(audio_path: str, out_dir: Path, lang: str):
    # Проверяем наличие кэшированных результатов
    cached_result = get_cached_result(audio_path, "asr")
    if cached_result:
        print("[ASR] Используем кэшированный результат распознавания")
        return cached_result
        
    from faster_whisper import WhisperModel
    import torch

    # Выбор устройства
    device = "cpu"
    if PREFER_GPU_ASR and (not FORCE_CPU_ASR) and torch.cuda.is_available():
        device = "cuda"

    model_name   = ASR_MODEL_GPU if device=="cuda" else ASR_MODEL_CPU
    compute_type = ASR_COMPUTE_TYPE_GPU if device=="cuda" else "int8"
    print(f"[ASR] device={device} model={model_name} compute_type={compute_type} chunk={ASR_CHUNK_LEN_SEC}s")

    # Ограничим треды, чтобы не дёргать планировщик
    for k in ["OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"]:
        os.environ[k] = "1"

    # Загрузка модели
    t0=time.time()
    try:
        model = WhisperModel(model_name, device=device, compute_type=compute_type, download_root="./models", local_files_only=False)
    except Exception as e:
        print(f"[ASR] init failed on {device}: {e}\n→ fallback to CPU {ASR_MODEL_CPU}/int8")
        try:
            model = WhisperModel(ASR_MODEL_CPU, device="cpu", compute_type="int8", download_root="./models", local_files_only=False)
            device="cpu"
        except Exception as e2:
            print(f"[ASR] critical error: {e2}\n→ trying offline openai/whisper-tiny")
            model = WhisperModel("openai/whisper-tiny", device="cpu", compute_type="int8", download_root="./models", local_files_only=False)
            device="cpu"

    print(f"[ASR] model loaded in {time.time()-t0:.1f}s")

    res = {"language": lang, "segments": []}
    srt_path  = str(out_dir/"fw_only.srt")
    json_path = str(out_dir/"fw_only.json")

    # Транскрипция небольшими чанками (снижает пиковую память)
    t1=time.time()
    print("[ASR] transcribing…")
    i=0
    with open(srt_path, "w", encoding="utf-8") as S:
        try:
            segments, info = model.transcribe(
                audio_path,
                language=lang,
                vad_filter=False,             # оставляем False, чтобы не тянуть torch-силеро
                beam_size=ASR_BEAM_SIZE,
                chunk_length=ASR_CHUNK_LEN_SEC,
                word_timestamps=False
            )
            for seg in segments:
                i += 1
                res["segments"].append({"start": seg.start, "end": seg.end, "text": seg.text})
                S.write(f"{i}\n{fmt_ts(seg.start)} --> {fmt_ts(seg.end)}\n{seg.text.strip()}\n\n")
                if i % 20 == 0:
                    print(f"[ASR] {i} segments…")
                    
                # Регулярно освобождаем память
                if i % 100 == 0:
                    gc.collect()
                    if device=="cuda":
                        torch.cuda.empty_cache()
        except Exception as e:
            print(f"[ASR ERROR] Ошибка при распознавании: {e}")
            raise

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    # Сохраняем в кэш
    save_cache(audio_path, "asr", res)

    # Очищаем память
    if device=="cuda":
        torch.cuda.empty_cache()
    gc.collect()

    print(f"[ASR] done: {i} segments in {time.time()-t1:.1f}s")
    print("ASR JSON:", json_path)
    print("ASR SRT :", srt_path)
    return res

res = run_asr(AUDIO, OUT_ASR, LANG)

# Функция для проверки пропусков в транскрипции
def check_for_gaps(segments, min_gap_sec=1.0):
    gaps = []
    for i in range(1, len(segments)):
        prev_end = segments[i-1].get('end', 0)
        curr_start = segments[i].get('start', 0)
        gap = curr_start - prev_end
        if gap > min_gap_sec:
            gaps.append({
                'prev_segment': i-1,
                'next_segment': i,
                'prev_end': prev_end,
                'next_start': curr_start,
                'gap_duration': gap,
                'prev_text': segments[i-1].get('text', ''),
                'next_text': segments[i].get('text', '')
            })
    return gaps

# Функция для создания текста, сгруппированного по говорящим
def create_speaker_grouped_text(segments):
    speakers = {}
    
    # Сначала группируем по спикерам
    for seg in segments:
        speaker = seg.get('speaker', 'SPEAKER_UNKNOWN')
        if speaker not in speakers:
            speakers[speaker] = []
        speakers[speaker].append({
            'start': seg.get('start', 0),
            'end': seg.get('end', 0),
            'text': seg.get('text', '').strip()
        })
    
    # Сортируем спикеров для консистентного вывода
    sorted_speakers = sorted(speakers.keys())
    
    # Формируем результат
    result = []
    for speaker in sorted_speakers:
        speaker_segments = speakers[speaker]
        # Сортируем сегменты этого спикера по времени
        speaker_segments.sort(key=lambda x: x['start'])
        speaker_text = [seg['text'] for seg in speaker_segments if seg['text'].strip()]
        if speaker_text:  # Если есть непустые сегменты
            result.append(f"\n{speaker}\n{' '.join(speaker_text)}\n")
    
    return ''.join(result)

# Функция для записи SRT файлов
def write_srt(segs, path, with_speaker=False):
    with open(path,"w",encoding="utf-8") as f:
        for idx, s in enumerate(segs,1):
            text = (s.get("text") or "").strip()
            spk  = s.get("speaker")
            if with_speaker and spk: text = f"[{spk}] {text}"
            f.write(f"{idx}\n{fmt_ts(s.get('start',0))} --> {fmt_ts(s.get('end',0))}\n{text}\n\n")

if not DIARIZE:
    # Если диаризация не нужна → финализируем тут
    
    # Добавляем дефолтного спикера, чтобы работала группировка
    for segment in res["segments"]:
        segment["speaker"] = "SPEAKER_DEFAULT"
    
    base = OUT_DIR/"final"
    # Сохраняем JSON с результатами
    with open(str(base)+".json","w",encoding="utf-8") as f: json.dump(res, f, ensure_ascii=False, indent=2)
    
    # Сохраняем SRT файлы
    write_srt(res["segments"], str(base)+".srt", False)
    write_srt(res["segments"], str(base)+"_with_speakers.srt", True)
    
    # Создаем и сохраняем группировку по спикерам
    grouped_text = create_speaker_grouped_text(res["segments"])
    with open(str(base)+"_grouped_by_speakers.txt", "w", encoding="utf-8") as f:
        f.write(grouped_text)
    
    # Проверяем на пропуски в транскрипции
    gaps = check_for_gaps(res["segments"], min_gap_sec=1.0)
    if gaps:
        print("\n⚠️ ВНИМАНИЕ: Обнаружены пропуски в транскрипции:")
        for i, gap in enumerate(gaps):
            print(f"  Пропуск {i+1}: {gap['gap_duration']:.2f} сек между {fmt_ts(gap['prev_end'])} и {fmt_ts(gap['next_start'])}")
            print(f"     До: '{gap['prev_text'][:50]}...'")
            print(f"     После: '{gap['next_text'][:50]}...'")
        
        # Сохраняем информацию о пропусках в файл
        with open(str(base)+"_gaps.json", "w", encoding="utf-8") as f:
            json.dump({"gaps": gaps}, f, ensure_ascii=False, indent=2)
    
    print("\n=== DONE ===")
    print("ASR  →", OUT_ASR)
    print("FINAL→", OUT_DIR)
    for p in [f"{base}.json", f"{base}.srt", f"{base}_with_speakers.srt", f"{base}_grouped_by_speakers.txt"]:
        print("  -", p)
        
    raise SystemExit
else:
    # ---------- ALIGN + DIAR ----------
    import whisperx
    print("[ALIGN] loading align model on CPU…")
    t0=time.time()
    audio_arr = whisperx.load_audio(AUDIO)
    align_model, meta = whisperx.load_align_model(language_code=res.get("language", LANG), device="cpu")
    print(f"[ALIGN] model loaded in {time.time()-t0:.1f}s")

    print("[ALIGN] aligning segments…")
    t1=time.time()
    aligned = whisperx.align(res["segments"], align_model, meta, audio_arr, "cpu", return_char_alignments=False)
    del align_model; gc.collect()
    print(f"[ALIGN] done in {time.time()-t1:.1f}s")

    def diarize_with_pyannote(aligned_obj):
        from whisperx.diarize import DiarizationPipeline, assign_word_speakers
        try:
            assert HF_TOKEN.startswith("hf_") and HF_TOKEN.isascii(), (
                "HF_TOKEN пуст/не ASCII. Для pyannote нужно принять доступ и задать токен."
            )
            print("[DIAR] pyannote (CPU)…")
            t0=time.time()
            dia = DiarizationPipeline(use_auth_token=HF_TOKEN, device="cpu")
            dsegs = dia(AUDIO, min_speakers=NUM_SPEAKERS, max_speakers=NUM_SPEAKERS)
            out  = assign_word_speakers(dsegs, aligned_obj)
            print(f"[DIAR] pyannote done in {time.time()-t0:.1f}s")
            return out
        except Exception as e:
            print(f"[DIAR ERROR] Ошибка при диаризации с PyAnnote: {e}")
            raise

    def diarize_open_fallback(aligned_or_res):
        print("[DIAR] fallback: Silero+ECAPA+KMeans…")
        pip_install(["speechbrain","scikit-learn","librosa","soundfile"])
        import numpy as np, librosa, torch
        from sklearn.cluster import KMeans
        from speechbrain.inference.speaker import EncoderClassifier

        wav, sr = librosa.load(AUDIO, sr=16000, mono=True)
        wav_t = torch.from_numpy(wav).float()
        vad_model, utils = torch.hub.load('snakers4/silero-vad','silero_vad',trust_repo=True)
        get_speech_timestamps = utils[0]
        speech_ts = get_speech_timestamps(wav_t, vad_model, sampling_rate=16000)
        if not speech_ts: raise RuntimeError("VAD не нашёл речи.")

        win, hop = int(1.5*16000), int(0.75*16000)
        windows=[]
        for ts_ in speech_ts:
            s,e=ts_['start'], ts_['end']; cur=s
            while cur<e:
                w0,w1=cur, min(e,cur+win)
                if w1-w0>=int(0.8*16000): windows.append((w0,w1))
                cur += hop

        clf = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
        embs=[]
        for s,e in windows:
            seg = torch.tensor(wav[s:e]).unsqueeze(0)
            with torch.no_grad(): emb = clf.encode_batch(seg).squeeze().cpu().numpy()
            embs.append(emb)
        embs = np.asarray(embs);
        km = KMeans(n_clusters=NUM_SPEAKERS, random_state=0, n_init=10).fit(embs)
        labels = km.labels_

        segments_spk=[]
        for idx,(s,e) in enumerate(windows):
            spk=int(labels[idx])
            if segments_spk and segments_spk[-1]['spk']==spk and s - segments_spk[-1]['end'] <= int(0.5*16000):
                segments_spk[-1]['end']=e
            else:
                segments_spk.append({'start':s,'end':e,'spk':spk})

        def maj_spk(st,en):
            s0,s1=int(st*16000), int(en*16000)
            votes={k:0 for k in range(NUM_SPEAKERS)}
            for d in segments_spk:
                a0,a1=d['start'],d['end']
                inter=max(0, min(s1,a1)-max(s0,a0))
                if inter>0: votes[d['spk']]+=inter
            if max(votes.values())==0: return None
            return max(votes, key=votes.get)

        final={"segments":[]}
        for seg in (aligned_or_res.get("segments") or []):
            k=maj_spk(seg["start"], seg["end"])
            spk=f"SPEAKER_{k:02d}" if k is not None else "SPEAKER_XX"
            final["segments"].append({**seg,"speaker":spk})
        return final

    # Проверяем наличие кэшированной диаризации
    cached_diar = get_cached_result(AUDIO, "diarization")
    if cached_diar:
        print("[DIAR] Используем кэшированный результат диаризации")
        final = cached_diar
    else:
        try:
            final = diarize_with_pyannote(aligned)
        except Exception as e:
            print(f"[DIAR] pyannote unavailable → {e}\n→ using fallback.")
            final = diarize_open_fallback(aligned)
        
        # Сохраняем результат диаризации в кэш
        save_cache(AUDIO, "diarization", final)

    # Функция для проверки пропусков в транскрипции
    def check_for_gaps(segments, min_gap_sec=1.0):
        gaps = []
        for i in range(1, len(segments)):
            prev_end = segments[i-1].get('end', 0)
            curr_start = segments[i].get('start', 0)
            gap = curr_start - prev_end
            if gap > min_gap_sec:
                gaps.append({
                    'prev_segment': i-1,
                    'next_segment': i,
                    'prev_end': prev_end,
                    'next_start': curr_start,
                    'gap_duration': gap,
                    'prev_text': segments[i-1].get('text', ''),
                    'next_text': segments[i].get('text', '')
                })
        return gaps
    
    # Функция для создания текста, сгруппированного по говорящим
    def create_speaker_grouped_text(segments):
        speakers = {}
        
        # Сначала группируем по спикерам
        for seg in segments:
            speaker = seg.get('speaker', 'UNKNOWN')
            if speaker not in speakers:
                speakers[speaker] = []
            speakers[speaker].append({
                'start': seg.get('start', 0),
                'end': seg.get('end', 0),
                'text': seg.get('text', '').strip()
            })
        
        # Сортируем спикеров для консистентного вывода
        sorted_speakers = sorted(speakers.keys())
        
        # Формируем результат
        result = []
        for speaker in sorted_speakers:
            speaker_segments = speakers[speaker]
            # Сортируем сегменты этого спикера по времени
            speaker_segments.sort(key=lambda x: x['start'])
            speaker_text = [seg['text'] for seg in speaker_segments if seg['text'].strip()]
            if speaker_text:  # Если есть непустые сегменты
                result.append(f"\n{speaker}\n{' '.join(speaker_text)}\n")
        
        return ''.join(result)
    
    # ---------- Сохранение ----------
    BASE = OUT_DIR/"final"
    
    # Сохраняем JSON с результатами
    with open(str(BASE)+".json","w",encoding="utf-8") as f: 
        json.dump(final, f, ensure_ascii=False, indent=2)
    
    # Сохраняем SRT файлы
    write_srt(final["segments"], str(BASE)+".srt", False)
    write_srt(final["segments"], str(BASE)+"_with_speakers.srt", True)
    
    # Создаем и сохраняем группировку по спикерам
    grouped_text = create_speaker_grouped_text(final["segments"])
    with open(str(BASE)+"_grouped_by_speakers.txt", "w", encoding="utf-8") as f:
        f.write(grouped_text)
    
    # Проверяем на пропуски в транскрипции
    gaps = check_for_gaps(final["segments"], min_gap_sec=1.0)
    if gaps:
        print("\n⚠️ ВНИМАНИЕ: Обнаружены пропуски в транскрипции:")
        for i, gap in enumerate(gaps):
            print(f"  Пропуск {i+1}: {gap['gap_duration']:.2f} сек между {fmt_ts(gap['prev_end'])} и {fmt_ts(gap['next_start'])}")
            print(f"     До: '{gap['prev_text'][:50]}...'")
            print(f"     После: '{gap['next_text'][:50]}...'")
        
        # Сохраняем информацию о пропусках в файл
        with open(str(BASE)+"_gaps.json", "w", encoding="utf-8") as f:
            json.dump({"gaps": gaps}, f, ensure_ascii=False, indent=2)
    
    print("\n=== DONE ===")
    print("ASR  →", OUT_ASR)
    print("FINAL→", OUT_DIR)
    for p in [f"{BASE}.json", f"{BASE}.srt", f"{BASE}_with_speakers.srt", f"{BASE}_grouped_by_speakers.txt"]:
        print("  -", p)
