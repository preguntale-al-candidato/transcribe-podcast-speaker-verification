# 🧉 Transcripción de Videos de Youtube con WhisperX y pyannote-audio

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

### 2. El código
Ese pedazo de código que me pasaste hace todo: baja el audio del video, lo transcribe, reconoce las voces, y te guarda todo en un archivo. Una masa.

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