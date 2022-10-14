import csv
import pandas as pd
import numpy as np
from Text_Based_SD.data.TranscriptProcess import *


def removeNan(row):
  res = []
  for i in row:
    if float(i) >= -1 and float(i) != 0:
      res.append(float(i))
  return res
  
def mapSpeaker(gt_tokens: list[Token], hyp_tokens: list[Token]):
  gt_spk1 = gt_tokens[0].spk_id
  for token in gt_tokens:
    if token.spk_id != gt_spk1:
      gt_spk2 = token.spk_id
      break
  hyp_spk1 = hyp_tokens[0].spk_id
  for token in hyp_tokens:
    if token.spk_id != hyp_spk1:
      hyp_spk2 = token.spk_id
      break
  
  return {gt_spk1:hyp_spk1, gt_spk2:hyp_spk2}

def get_speaker_list(token_list):
  spk1 = token_list[0].spk_id
  for token in token_list:
    if token.spk_id != spk1:
      spk2 = token.spk_id
      return (spk1, spk2)


class Eval_3d:
  def __init__(self, file_code, type) -> None:
    self.file_code = file_code
    if type == "Amazon":
      self.resultPath = f"ResultAmazon/Result3D/{self.file_code}_result_amazon.csv"
      # self.transcriptPath = f"../data/CallHome_eval/amazon/{self.file_code}.json"
      self.hyp_tokens = Amazon(f"../data/CallHome_eval/amazon/{self.file_code}.json").get_token_list()
      
    self.spk1_align, self.spk2_align = self.read3dResult()
    self.gt_tokens = CallHome(f"../data/CallHome_eval/transcripts/{self.file_code}.cha").get_token_list()
    self.gt_spk_ids = get_speaker_list(self.gt_tokens)
    self.hyp_spk_ids = get_speaker_list(self.hyp_tokens)
    self.spk_map = mapSpeaker(self.gt_tokens, self.hyp_tokens)

  
  def read3dResult(self):
    df = pd.read_csv(self.resultPath,header=None).fillna(-10)
    spk1_align_raw = df.values[3, 1:]
    spk2_align_raw = df.values[4, 1:]
    spk1_align = removeNan(spk1_align_raw)
    spk2_align = removeNan(spk2_align_raw)
    return spk1_align, spk2_align

  def calculate(self):
    error_count, correct_count, mismatch = 0, 0, 0
    spk1_error, spk2_error = [], []
    for i in range(len(self.spk1_align)):
      index = int(self.spk1_align[i]) - 1
      if index < 0:
        mismatch += 1
        continue
      if self.hyp_tokens[index].spk_id == self.spk_map["A"]:
        correct_count += 1
      else:
        error_count += 1
        spk1_error.append(i)
    
    for i in range(len(self.spk2_align)):
      index = int(self.spk2_align[i]) - 1
      if index < 0:
        mismatch += 1
        continue
      if self.hyp_tokens[index].spk_id == self.spk_map["B"]:
        correct_count += 1
      else:
        error_count += 1
        spk2_error.append(i)
    
    print("Error: ", error_count)
    print("Correct: ", correct_count)
    print("Mismatch: ", mismatch)
    print("Token Count: ", len(self.gt_tokens))
    print("Correct Rate: ", correct_count/len(self.gt_tokens))
    print("Error Rate: ", error_count/len(self.gt_tokens))
    return error_count, correct_count, mismatch
    


class Eval_2d():
  def __init__(self, file_code) -> None:
    self.file_code = file_code
    self.resultPath = f"ResultAmazon/Result2DCombined/{file_code}_result_amazon.csv"
    self.gt_to_hyp, self.hyp_to_gt = self.read2dResult()
    
    self.gt_tokens = CallHome(f"../data/CallHome_eval/transcripts/{self.file_code}.cha").get_token_list()
    self.hyp_tokens = Amazon(f"../data/CallHome_eval/amazon/{self.file_code}.json").get_token_list()
    self.spk_map = mapSpeaker(self.gt_tokens, self.hyp_tokens)


  def read2dResult(self):
    df = pd.read_csv(self.resultPath,header=None).fillna(-10)
    gt_align_hyp_raw = df.values[2, 1:]
    hyp_align_gt_raw = df.values[3, 1:]
    gt_align_hyp = removeNan(gt_align_hyp_raw)
    hyp_align_gt = removeNan(hyp_align_gt_raw)
    return gt_align_hyp, hyp_align_gt
  
  def calculate(self):
    correct_count, error_count, mismatch = 0,0,0
    for i in range(len(self.gt_to_hyp)):
      index = i-1
      hyp_index = int(self.gt_to_hyp[i]) - 1
      gt_spk = self.gt_tokens[index].spk_id
      hyp_spk = self.hyp_tokens[hyp_index].spk_id
      if hyp_index == -1:
        mismatch += 1
        continue
      if hyp_spk == self.spk_map[gt_spk]:
        correct_count += 1
      else:
        error_count += 1
        
    print("Error: ", error_count)
    print("Correct: ", correct_count)
    print("Mismatch: ", mismatch)
    print("Token Count: ", len(self.gt_tokens))
    print("Correct Rate: ", correct_count/len(self.gt_tokens))
    print("Error Rate: ", error_count/len(self.gt_tokens))
    return error_count, correct_count, mismatch



def output_3d_eval_csv():
  header = ["file", "error_rate", "correct_rate (recall)", "error_count", "correct_count", "mismatch", "total_token"]
  pool = ["4074", "4315", "4093", "4247", "4325", "4335", "4571", "4595", "4660", "4290"]
  with open("ResultAmazon/Amazon_3D_eval.csv", 'w') as file:
    output = csv.writer(file)
    output.writerow(header)
    for file_code in pool:
      print("======= ", file_code)
      eval = Eval_3d(file_code=file_code, type="Amazon")
      error, correct, mismatch = eval.calculate()
      token_len = len(eval.gt_tokens)
      row = [file_code, error/token_len, correct/token_len, error, correct, mismatch, token_len]
      output.writerow(row)

def output_2d_eval_csv():
  header = ["file", "error_rate", "correct_rate (recall)", "error_count", "correct_count", "mismatch", "total_token"]
  pool = ["4074", "4315", "4093", "4247", "4325", "4335", "4571", "4595", "4660", "4290"]
  with open("ResultAmazon/Amazon_2D_eval.csv", 'w') as file:
    output = csv.writer(file)
    output.writerow(header)
    for file_code in pool:
      print("======= ", file_code)
      eval = Eval_2d(file_code=file_code)
      error, correct, mismatch = eval.calculate()
      token_len = len(eval.gt_tokens)
      row = [file_code, error/token_len, correct/token_len, error, correct, mismatch, token_len]
      output.writerow(row)


if __name__ == "__main__":
  # file_code = 4335
  # eval = Eval_3d(file_code=file_code, type="Amazon")
  # eval.calculate()
  # print(eval.gt_spk_ids)
  # print(eval.hyp_spk_ids)
  # print(eval.spk_map)
  
  # file_code = 4074
  # eval = Eval_2d(file_code=file_code)
  # eval.calculate()
  
  # output_3d_eval_csv()
  
  output_2d_eval_csv()
