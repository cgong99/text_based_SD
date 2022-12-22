from jiwer import wer
import csv
from Text_Based_SD.data.TranscriptProcess import Amazon, CallHome, RevAI



def getAllWER(type: str, output_file:str):
  # "ResultRevAI/Rev_3D_DER.csv"
  # header = ["file",  "Precision_DER", "Recall_DER"]
  header = ["file", "DER"]
  pool = ["4074", "4315", "4093", "4247", "4325", "4335", "4571", "4595"]
  with open(output_file, 'w') as file:
    output = csv.writer(file)
    output.writerow(header)
    for file_code in pool:
      print(file_code)
      gt_transcript = CallHome(f"../data/CallHome_eval/transcripts/{file_code}.cha")
      if type == "Amazon":
        hyp_transcript = Amazon(f"../data/CallHome_eval/amazon/{file_code}.json")
      else:
        hyp_transcript = RevAI(f"../data/CallHome_eval/rev/{file_code}_cut.json")
      gt_str = ""
      for a in gt_transcript.get_file_annotation(with_utterances=True):
        utt=a[3]
        gt_str = gt_str + " " + utt

      hyp_str = ""
      for a in hyp_transcript.get_utterances_by_spkID():
        utt = a[1]
        hyp_str = hyp_str + " " + utt


      error = wer(gt_str, hyp_str)
      row = [file_code, error]
      output.writerow(row)
      
if __name__ == '__main__':
  gt = "hello I am from America"
  hyp = "Hi I'm from America"

  gt = ["hello", "I am from America"]
  hyp = ["Hi I am", "from America"]

  file_code = 4074
  gt_transcript = CallHome(f"../data/CallHome_eval/transcripts/{file_code}.cha")
  hyp_transcript = Amazon(f"../data/CallHome_eval/amazon/{file_code}.json")
  gt_str = ""
  for a in gt_transcript.get_file_annotation(with_utterances=True):
    utt=a[3]
    gt_str = gt_str + " " + utt

  hyp_str = ""
  for a in hyp_transcript.get_utterances_by_spkID():
    utt = a[1]
    hyp_str = hyp_str + " " + utt

  print(len(gt_str))
  print(len(hyp_str))
  error = wer(gt_str, hyp_str)
  print(error)
  

  # getAllWER('Amazon',  "ResultAmazon/Amazon_WER.csv")
  getAllWER('Rev',  "ResultRevAI/Rev_WER.csv")