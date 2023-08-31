# 🧉 Transcripción de videos e identificación de speakers

Che, bienvenidos a este laburo de transcripción automática de videos de Youtube. Además, no sólo transcribimos, ¡sino que identificamos a los que charlan en el video!

## 📜 Descripción

Con este chiche, podés transcribir cualquier video de Youtube y, arriba de eso, diferenciar las voces de los que hablan. Se re viene para entender quién tira la posta en un debate, entrevista, o charla.

## 🛠 Lo que necesitás

- **Python**: De última versión, no me vengas con algo viejo.
- **yt-dlp**: Para bajar el audio de los videos de Youtube.
- **whisperX**: Para transcribir el audio.
- **pyannote-audio**: Para reconocer las voces y diferenciar a los hablantes.

## 📌 Pasos para arrancar con todo

### 1. Instalar las cosas

Primero, instala todo lo que te dije arriba. Acá te dejo cómo:

```bash
pip install git+https://github.com/m-bain/whisperx.git
python3 -m pip install -U yt-dlp
```

También podés instalar directamente el `environment.yml`.

```bash
$ conda env create -f environment.yml
$ conda activate machinelearnear-dev
```

### 2. El código
Ese pedazo de código (`scripts/run_process.py`) que me pasaste hace todo: baja el audio del video, lo transcribe, reconoce las voces, y te guarda todo en un archivo. Una masa.

Acá esta la magia:

```python
# download audio from youtube url
download_audio(each.url, temp_dir)

# transcribe and perform speaker diarization
audio = whisperx.load_audio(target_audio)
result = model.transcribe(audio, batch_size=batch_size, language='es')

# align whisper output
model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

# assign speaker labels
diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
diarize_segments = diarize_model(audio)
result = whisperx.assign_word_speakers(diarize_segments, result)

# start audio pipeline
audio_pipe = Audio(sample_rate=16000, mono="downmix")
reference_wav_fname = f'{ref_audio_dir}/audio_{reference[each.channel_name]}.wav'

# extract embeddings for a speaker speaking between t=0 and t=10s
segment_reference = Segment(1., 10.)
waveform_reference, sample_rate = audio_pipe.crop(reference_wav_fname, segment_reference)
embedding_reference = embeddings_model(waveform_reference[None])

# add cosine dist to each segment in the results
for segment in result['segments']:
    # extract embedding for a speaker speaking between t=Xs and t=Ys
    speaker_target = Segment(segment['start'], segment['end'])
    waveform_target, sample_rate = audio_pipe.crop(target_audio, speaker_target)
    embedding_target = embeddings_model(waveform_target[None])

    # compare embeddings using "cosine" distance
    distance = cdist(embedding_reference, embedding_target, metric="cosine")
    segment['cosine_dist'] = distance[0][0]

    # save back the info to the dict
    segment['is_candidate'] = True if distance[0][0] <= cdist_threshold else False

# save result dictionary to local disk
save_dict_to_disk(result, each, reference, output_dir, filename=each.id)

# delete temp folders
shutil.rmtree(temp_dir)
```

### 3. Usar el script

Poné en la terminal:

```bash
python scripts/run_process.py --hf_token TU_HF_TOKEN --data_dir RUTA_DEL_DATA --ref_audio_dir RUTA_DEL_AUDIO_REFERENCIA --temp_dir RUTA_TEMPORAL --output_dir RUTA_DE_SALIDA
```

Reemplazá las palabras en mayúscula por tus datos.

## 💡 Algunos piques

- Seguí los pasos como te los di, así no hay lío.
- Si te encontrás con algún error, revisá que las rutas de los archivos estén bien y que hayas puesto todos los argumentos necesarios.
- No te olvides de poner tu token de Hugging Face cuando uses el script.

## 🤝 ¿Querés sumarte?

Si te gusta lo que hicimos y tenés alguna idea para mejorarlo, ¡dale, unite! Mandá lo que hiciste y vemos cómo lo metemos.

## 📬 ¿Dudas?

Si algo no te cierra o necesitás una mano, escribinos. Estamos para ayudarte.
