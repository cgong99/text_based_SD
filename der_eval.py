import simpleder
from TranscriptProcess import *
import os
from pyannote.core import Segment, Timeline, Annotation
from pyannote.metrics.diarization import DiarizationErrorRate

if __name__ == "__main__":
  name1 = "CallHome_eval/transcripts/4093.cha"
  name2 = "CallHome_eval/rev/4093_cut.json"
  ref_path = "CallHome_eval/transcripts/"
  # hyp_path = "CallHome_eval/rev/"
  hyp_path = "CallHome_eval/amazon/"

  ref_files = os.listdir(ref_path)
  hyp_files = os.listdir(hyp_path)
  
  # ref = CallHome(name1).get_file_annotation(True)
  # hyp = RevAI(name2).get_file_annotation()
  # error = simpleder.DER(ref, hyp)
  # print("DER={:.3f}".format(error))

  score = {}
  for file in ref_files:
    name = file[:file.find(".")]
    for hyp_file in hyp_files:
      if name in hyp_file:
          ref = CallHome(ref_path+file).get_file_annotation(start_with_zero=True, allow_overlap=False)
          # hyp = RevAI(hyp_path+hyp_file).get_file_annotation()
          hyp = Amazon(hyp_path+hyp_file).get_file_annotation()
          error = simpleder.DER(ref, hyp)
          score[name] = error
          print(name, "DER={:.3f}".format(error))
  print(sum(score.values())/len(score.keys()))

# Rev
# 4660 DER=0.372
# 4315 DER=0.534
# 4074 DER=0.229
# 4571 DER=0.324
# 4595 DER=0.664
# 4093 DER=0.679
# 4247 DER=0.524
# 4290 DER=1.716
# 4325 DER=0.476
# 4335 DER=0.522
# 0.6039822725596379

# Amazon
# 4660 DER=0.539
# 4315 DER=0.212
# 4074 DER=0.161
# 4571 DER=0.284
# 4595 DER=0.268
# 4093 DER=0.202
# 4247 DER=0.169
# 4290 DER=0.258
# 4325 DER=0.223
# 4335 DER=0.213
# 0.2527870975211899

