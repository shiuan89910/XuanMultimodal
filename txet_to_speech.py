import getpass
import logging
import os
import time
import torch
import torchaudio
from pydub import AudioSegment
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts


proj_name = "XuanMultimodal"
try:
  user_name = os.getlogin()
  root_path = f"C:\\Users\\{user_name}\\{proj_name}"
except:
  root_path = f"/{proj_name}"


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# 將 M4A 格式音頻轉換為 WAV 格式
def convert_m4a_to_wav(input_file, output_file):
  try:
    audio = AudioSegment.from_file(input_file, format="m4a")
    audio.export(output_file, format="wav")
  except Exception as e:
    logging.error(f"convert_m4a_to_wav() ERR: {e}")


# 從預訓練模型進行文字到語音的轉換
def tts_from_pretrained(**kwargs):
    try:
      config = XttsConfig()
      config.load_json(f'{root_path}/{kwargs["model_folder"]}/config.json')
      model = Xtts.init_from_config(config)
      model.load_checkpoint(config, checkpoint_dir=f'{root_path}/{kwargs["model_folder"]}/', eval=True)
      model.cuda()
      return model, config
    except Exception as e:
      logging.error(f"tts_from_pretrained() ERR: {e}")
      return None, None


# 文本轉語音生成
def tts_gen(model, config, text, output_path, **kwargs):
  try:
    if not os.path.exists(f'{root_path}/{kwargs["split_folder"]}'):
      os.makedirs(f'{root_path}/{kwargs["split_folder"]}')
    text_parts = split_text_by_punctuation_and_length(text, **kwargs)
    for i, part in enumerate(text_parts):
      outputs = model.synthesize(
        part,
        config,
        speaker_wav=f'{root_path}/{kwargs["model_folder"]}/samples/{kwargs["voice_name"]}.wav',
        gpt_cond_len=10,
        language=kwargs["language"],
        )
      wav_path = f'{root_path}/{kwargs["split_folder"]}/{i}.wav'
      torchaudio.save(wav_path, torch.tensor(outputs["wav"]).unsqueeze(0), 24000)
      if len(text_parts) == i + 1:
        while not os.path.exists(wav_path):
          time.sleep(0.001)
    combine_wav_files(output_path, **kwargs)
  except Exception as e:
    logging.error(f"tts_gen() ERR: {e}")


# 根據標點和長度分割文本
def split_text_by_punctuation_and_length(text, **kwargs):
  try:
    punctuation_marks = "。！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
    punctuation_marks += "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    segments = []
    start = 0
    for i, char in enumerate(text):
      if char in punctuation_marks:
        segment = text[start:i + 1].strip()
        if len(segment) > kwargs["text_max_length"]:
          segments.extend([segment[j:j+kwargs["text_max_length"]] for j in range(0, len(segment), kwargs["text_max_length"])])
        else:
          segments.append(segment)
        start = i + 1
    if start < len(text):
      remaining_segment = text[start:].strip()
      if len(remaining_segment) > kwargs["text_max_length"]:
        segments.extend([remaining_segment[j:j+kwargs["text_max_length"]] for j in range(0, len(remaining_segment), kwargs["text_max_length"])])
      else:
        segments.append(remaining_segment)
    return segments
  except Exception as e:
    logging.error(f"split_text_by_punctuation_and_length() ERR: {e}")
    return []


# 合併 WAV 文件
def combine_wav_files(output_path, **kwargs):
  try:
    files = [f for f in os.listdir(f'{root_path}/{kwargs["split_folder"]}') if f.endswith(".wav")]
    files = [os.path.join(f'{root_path}/{kwargs["split_folder"]}', f) for f in files]
    combined = AudioSegment.empty()
    for wav_file in files:
      sound = AudioSegment.from_wav(wav_file)
      combined += sound
    combined.export(output_path, format="wav")
  except Exception as e:
    logging.error(f"combine_wav_files() ERR: {e}")


# 讀取文本文件
def read_text_file(file_path, encoding="utf-8"):
  try:
    with open(file_path, "r", encoding=encoding) as file:
      return file.read()
  except Exception as e:
    logging.error(f"read_text_file() ERR: {e}")
    return ""


# 測試 TTS 功能
def test_tts():
  param_gen = {
    "model_folder": "XTTS-v2",
    "text_max_length": 10,
    "voice_name": "samples_zh-cn-sample",
    "language": "zh-cn",
    "split_folder": "Temp",
    }
  data_path = f"{root_path}/test"
  text = read_text_file(data_path + ".txt")
  model, config = tts_from_pretrained(**param_gen)
  tts_gen(model, config, text, data_path + ".wav", **param_gen)


if __name__ == "__main__":
  test_tts()