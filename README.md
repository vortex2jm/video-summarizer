## Blog text generator

 - This script receives a video as input and follows the pipeline below:
 1. Extracts the .mp3 audio from video with a 16k bit rate, 12KHz sample rate, and 1 channel;
 2. Transcripts the audio with FasterWhisper model;
 3. Uses DeepSeek to create a summary with the transcripted audio;

 - Create an API Key in [https://openrouter.ai/](https://openrouter.ai/) to enjoy all features of the script.

### Requirements
```bash
sudo apt install ffmpeg
```

```bash
pip install -r requirements.txt
```

### Usage
```bash
python3 main.py <input.mp4>
```
