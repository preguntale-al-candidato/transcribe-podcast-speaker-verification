# 🧉 Transcripción de Videos de Youtube con Identificación de Agentes

¡Che, bienvenidos al proyecto de transcripción de videos de Youtube e identificación de diferentes agentes en la conversación! Este proyecto fue inspirado por el genial **machinelearnear**. Si te gusta el mundo de la inteligencia artificial y cómo puede ser aplicado en la vida real, estás en el lugar correcto.

## 📜 Descripción

Con este proyecto, no sólo podrás transcribir videos de Youtube automáticamente, sino también identificar y diferenciar las voces de los diferentes participantes en una conversación. Esto es útil para entender quién dice qué en un debate, entrevista, charla, entre otros.

## 🛠 Pre-requisitos

### Cuenta de Google y Google Drive
Necesitarás una cuenta de Google para poder montar y usar Google Drive como almacenamiento persistente.

### Carpeta compartida
Asegurate de tener acceso a esta carpeta compartida: ['preguntale-al-candidato'](https://drive.google.com/drive/folders/1HKcNUU_Ws8VJnlg5O4r8WUrbuHwu9P84?usp=sharing).

## 📌 Pasos para Instalar y Usar

### 1. Instalar Dependencias

Vamos a necesitar algunas librerías para que todo funcione correctamente:

```bash
pip install git+https://github.com/m-bain/whisperx.git
python3 -m pip install -U yt-dlp
```

### 2. Montar Google Drive

Una vez que hayas instalado las dependencias, es necesario montar Google Drive para acceder a los datos necesarios para el proyecto. Seguí las instrucciones en el código para hacerlo.

### 3. Librerías Utilizadas

Este proyecto hace uso de dos librerías principales:

- **whisperX**: Es una herramienta que permite transcribir audios y videos. Puedes conocer más [aquí](https://github.com/m-bain/whisperX).
- **pyannote-audio**: Es una herramienta de gran utilidad para la verificación y reconocimiento de hablantes en audios. Para entender su funcionamiento y cómo se integra en este proyecto, te recomiendo leer este [tutorial](https://github.com/pyannote/pyannote-audio/blob/develop/tutorials/speaker_verification.ipynb).

## 💡 Consejos

Si es la primera vez que trabajas con proyectos de este tipo, te recomiendo que sigas los pasos al pie de la letra. Cualquier cambio en el código puede generar errores o comportamientos inesperados. Si tienes dudas, consulta la documentación de las librerías mencionadas.

## 🤝 Contribuciones

Si te copás con el proyecto y tenés ideas para mejorarlo o expandirlo, ¡no dudes en colaborar! Mandá tu pull request y charlemos sobre cómo incorporar tus aportes.

## 📬 Contacto

Si tenés alguna pregunta, sugerencia o problema, no dudes en abrir un "issue" en este repositorio. Estamos para ayudarte.
