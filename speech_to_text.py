import getpass
import logging
import os
import torch
from datasets import load_dataset
from moviepy.editor import AudioFileClip
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


proj_name = "XuanMultimodal"
try:
   user_name = os.getlogin()
   root_path = f"C:\\Users\\{user_name}\\{proj_name}"
except:
   root_path = f"/{proj_name}"


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# 將 M4A 格式的音頻文件轉換為 MP3 格式
def convert_m4a_to_mp3(input_file, output_file):
   try:
      audio_clip = AudioFileClip(input_file)
      audio_clip.write_audiofile(output_file, codec="mp3")
   except Exception as e:
      logging.error(f"convert_m4a_to_mp3() ERR: {e}")


# 從預訓練模型進行語音到文字的轉換
def stt_from_pretrained(**kwargs):
   try:
      device = "cuda:0" if torch.cuda.is_available() else "cpu"
      torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
      model = AutoModelForSpeechSeq2Seq.from_pretrained(
         f'{root_path}/{kwargs["model_id"]}',
         torch_dtype=torch_dtype,
         low_cpu_mem_usage=kwargs["low_cpu_mem_usage"],
         use_safetensors=kwargs["use_safetensors"],
         use_flash_attention_2=kwargs["use_flash_attention_2"], # Flash Attention
         )
      model = model.to_bettertransformer() if kwargs["use_SPDA"] else model # Torch Scale-Product-Attention (SDPA)
      model.to(device)
      processor = AutoProcessor.from_pretrained(f'{root_path}/{kwargs["model_id"]}')
      pipe = pipeline(
         "automatic-speech-recognition",
         model=model,
         tokenizer=processor.tokenizer,
         feature_extractor=processor.feature_extractor,
         max_new_tokens=kwargs["max_new_tokens"],
         chunk_length_s=kwargs["chunk_length_s"],
         batch_size=kwargs["batch_size"],
         return_timestamps=kwargs["return_timestamps"],
         torch_dtype=torch_dtype,
         device=device,
         )  
      return pipe
   except Exception as e:
      logging.error(f"stt_from_pretrained() ERR: {e}")
      return None


# 使用預訓練的語音識別模型生成轉換結果
def stt_generate(data_src, pipe, **kwargs):
   try:
      result = pipe(
         data_src,
         return_timestamps=kwargs["return_timestamps"],
         generate_kwargs={"language": kwargs["language"], "task": kwargs["task"]},
         )
      return result[kwargs["ret_format"]]
   except Exception as e:
      logging.error(f"stt_generate() ERR: {e}")
      return None  


# 測試語音到文字的功能
def test_stt():
   file_path = f"{root_path}/test"
   convert_m4a_to_mp3(file_path + ".m4a", file_path + ".mp3")
   param_fp = {
      "model_id": "whisper-medium",
      "low_cpu_mem_usage": True,
      "use_safetensors": True,
      "use_flash_attention_2": False,
      "use_SPDA": False,
      "max_new_tokens": 128,
      "chunk_length_s": 30,
      "batch_size": 16,
      "return_timestamps": True, # return_timestamps=True or return_timestamps="word"
      }
   data_src = file_path + ".mp3" # load_dataset("distil-whisper/librispeech_long", "clean", split="validation")[0]["audio"]
   pipe = stt_from_pretrained(**param_fp)
   param_gen = {
      "return_timestamps": True, # return_timestamps=True or return_timestamps="word"
      "language": "english", # "language": "english" or None
      "task": "translate", # "task": "translate" or None
      "ret_format": "chunks", # "chunks" or "text"
      }
   ret = stt_generate(data_src, pipe, **param_gen)
   logging.info(ret)


if __name__ == "__main__":
   test_stt()