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
  

  # with open("./transcription.txt", "r") as file:
  #   transcription = file.read()

  # prompt_content = f"""
  #   Summarize the following content clearly and objectively, 
  #   highlighting the main points and key ideas from the video. 
  #   Keep the essence of what was said, but make it concise without 
  #   losing the important details. If necessary, organize the summary 
  #   in a structured way, separating the information into main topics. 
  #   Avoid including repetitions or irrelevant details. The content 
  #   to be summarized is the transcript below:
  #   {transcription}
  # """

  prompt_content = f"""
    Below is the full transcription of a video. Your task is to carefully analyze this content and generate a detailed yet concise summary. The goal is to turn this transcript into a well-structured, coherent text that highlights the main points covered in the video, including key arguments, conclusions, technical concepts, and any noteworthy or relevant information.

    Guidelines:
    Do not copy sentences directly from the transcript — rewrite them clearly and objectively.
    Use natural, fluent language, as if explaining the video to someone who hasn't watched it.
    If possible, organize the summary using headings or subheadings, based on the topics discussed in the video.
    Briefly define any technical terms to ensure clarity for a general audience.
    If the video presents opinions, arguments, or debates, explain who defends what (if identifiable), and summarize the pros and cons discussed.
    If it's an instructional video (e.g., a tutorial, lecture, or workshop), clearly highlight the steps taught, key tips, and practical takeaways.
    Finish with a general conclusion, summarizing the overall purpose of the video and who would benefit from watching it.
    The text must be in Portuguese

    Transcript:
    {transcription}
  """

  # consult AI
  response = prompt(TOKEN, prompt_content, MODEL)
  
  if response[0] == 200:
    write_output(TEXT_OUTPUT, response[1])
  else:
    print(f"Error: {response[1]}")
