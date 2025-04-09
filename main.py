import subprocess
import requests
import json
import sys
import os

from faster_whisper import WhisperModel
from dotenv import load_dotenv

load_dotenv()

#===============================================
def prompt(token: str, content: str, model: str):
  response = requests.post(
    url="https://openrouter.ai/api/v1/chat/completions",
    headers={
      "Authorization": f"Bearer {token}",
      "Content-Type": "application/json"
    },
    data=json.dumps({
      "model": f"{model}",
      "messages": [
        {
          "role": "user",
          "content": f"{content}"
        }
      ],
    })
  )
  response_json = response.json()
  
  if response.status_code == 200:
    return (200, response_json['choices'][0]['message']['content'])
  else:
    return (response.status_code, response_json)

#============================
def write_output(path, content):
  with open(path, 'w') as f:
    f.write(content)

#==============================
def extract_audio_from_mp4(input_path, output_path):
  command = f"ffmpeg -i {input_path} -vn -acodec libmp3lame -b:a 16k -ar 12000 -ac 1 {output_path}"
  try:
    subprocess.run(command, shell=True, check=True)
    return output_path
  except subprocess.CalledProcessError as e:
    print(f"Error: {e}")
    return None

#=================================
def transcript_audio_to_text(path):
  model_size = "small"
  model = WhisperModel(model_size, device="cpu", compute_type="int8")
  segments, info = model.transcribe(path, beam_size=5)
  
  text = ""
  print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
  
  for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
    text += f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}\n"
  
  return text


# entrypoint==============
if __name__ == "__main__":

  # const variables
  MODEL = "deepseek/deepseek-chat-v3-0324:free"
  TOKEN = os.getenv('TOKEN')
  OUTPUT_MP3 = "audio.mp3"
  TEXT_OUTPUT = "summary.md"

  # check args
  if len(sys.argv) < 2:
    print("Usage: python main.py <input_file>")
    exit(1)

  # extracting audio
  input_file = sys.argv[1]
  if not extract_audio_from_mp4(input_file, OUTPUT_MP3):
    exit(1)

  # transcripting audio
  transcription = transcript_audio_to_text(OUTPUT_MP3)
  write_output("transcription.txt", transcription)
  
  prompt_content = f"""
    Summarize the following content clearly and objectively, 
    highlighting the main points and key ideas from the video. 
    Keep the essence of what was said, but make it concise without 
    losing the important details. If necessary, organize the summary 
    in a structured way, separating the information into main topics. 
    Avoid including repetitions or irrelevant details. The content 
    to be summarized is the transcript below:
    {transcription}
  """

  # consult AI
  response = prompt(TOKEN, prompt_content, MODEL)
  
  if response[0] == 200:
    write_output(TEXT_OUTPUT, response[1])
  else:
    print(f"Error: {response[1]}")
