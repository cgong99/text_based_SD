from pyannote.core import Segment, Timeline, Annotation
from pyannote.metrics.diarization import DiarizationErrorRate
from Text_Based_SD.data.TranscriptProcess import CallHome, Amazon, RevAI
import csv
# file_code = 4074
# reference = Annotation()
# hypothesis = Annotation()
# for segment in CallHome(f"../data/CallHome_eval/transcripts/{file_code}.cha").get_file_annotation():
#     reference[Segment(segment[1], segment[2])] = segment[0]
# for segment in RevAI(f"../data/CallHome_eval/rev/{file_code}_cut.json").get_file_annotation():
#     hypothesis[Segment(segment[1], segment[2])] = segment[0]

# metric = DiarizationErrorRate()
# print(metric(reference, hypothesis, detailed=True))

def writeDERcsv(type:str, output_file:str):
  # "ResultRevAI/Rev_3D_DER.csv"
  # header = ["file",  "Precision_DER", "Recall_DER"]
  header = ["file", "DER"]
  pool = ["4074", "4315", "4093", "4247", "4325", "4335", "4571", "4595"]
  with open(output_file, 'w') as file:
    output = csv.writer(file)
    output.writerow(header)
    for file_code in pool:
      reference = Annotation()
      hypothesis = Annotation()
      for segment in CallHome(f"../data/CallHome_eval/transcripts/{file_code}.cha").get_file_annotation():
          reference[Segment(segment[1], segment[2])] = segment[0]
      for segment in RevAI(f"../data/CallHome_eval/rev/{file_code}_cut.json").get_file_annotation():
          hypothesis[Segment(segment[1], segment[2])] = segment[0]

      metric = DiarizationErrorRate()
      der = metric(reference, hypothesis)
      row = [file_code, der]
      output.writerow(row)
      

writeDERcsv("Rev", "ResultRevAI/Rev_audio_DER.csv")
  
  # with open(output_file, 'w') as file:
  #   output = csv.writer(file)
  #   output.writerow(header)
  #   for code in pool:
  #     print(code)
  #     gt_der, hyp_der = computeTwoDER(file_code=code, type=type)
  #     row = [code, hyp_der, gt_der]
  #     output.writerow(row)