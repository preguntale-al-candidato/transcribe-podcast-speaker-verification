# 🧉 Transcripción de Podcasts con Identificación de Speakers

¡Che, bienvenidos al repo! Acá vas a encontrar un proyecto que no sólo transcribe podcasts, sino que también identifica a los diferentes oradores. ¿Y sabés qué es lo mejor? ¡Podés extraer el texto de un solo orador!

| Colab | Info
| --- | --- |
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/machinelearnear/88bf703112e5acf87d1e4cac76772a72/whisperx-example-youtube.ipynb) | detect_custom_speaker_from_podcast

## Cómo empezar

1. **Definir Input** - Primero que nada, definí tu input.
2. **Instalación** - Antes de ponerte a laburar, instalá las cosas necesarias. No te preocupes, está todo en el código.
3. **Descarga del video desde YouTube** - ¿Tenés un video en YT que querés transcribir? ¡Dale, descargalo!
4. **Transcribir y hacer Speaker Diarization** - Una vez que tengas el audio, es hora de transcribirlo y detectar a los distintos oradores.
5. **Mergear Segmentos** - Basándote en los speakers identificados, podés mergear segmentos.
6. **Verificación de Speaker** - Si querés verificar un speaker en particular, tomá un clip de audio como baseline y dale para adelante.

## Código Inicial

Para darte una idea, acá tenés un adelanto del código:

```python
hf_token = "<aca-va-el-hf-token>" # Token de HuggingFace
youtube_video = 'URL_DEL_VIDEO' # Poné acá la URL de tu video
youtube_video_candidato = 'URL_DEL_VIDEO_CORTO' # Debe ser un video corto, de menos de 60 segundos

# Instalaciones necesarias
!pip install git+URL_DEL_REPO;
!python3 -m pip install -U yt-dlp;

# Descargar el audio del video
!python -m yt_dlp --output "audio.%(ext)s" --extract-audio --audio-format wav $youtube_video

# Transcribir y hacer Speaker Diarization
!whisperx audio.wav --hf_token $hf_token --model modelo --language es --align_model MODELO_A_USAR --diarize --min_speakers 2
