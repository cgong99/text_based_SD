import simpleder
from TranscriptProcess import *
import os
from pyannote.core import Segment, Timeline, Annotation
from pyannote.metrics.diarization import DiarizationErrorRate

if __name__ == "__main__":
  name1 = "CallHome_eval/transcripts/4093.cha"
  name2 = "CallHome_eval/rev/4093_cut.json"
  ref_path = "CallHome_eval/transcripts/"
  hyp_path = "CallHome_eval/rev/"

  ref_files = os.listdir(ref_path)
  hyp_files = os.listdir(hyp_path)
  
  # ref = CallHome(name1).get_file_annotation(True)
  # hyp = RevAI(name2).get_file_annotation()
  # error = simpleder.DER(ref, hyp)
  # print("DER={:.3f}".format(error))

  # for file in ref_files:
  #   name = file[:file.find(".")]
  #   for hyp_file in hyp_files:
  #     if name in hyp_file:
  #         ref = CallHome(ref_path+file).get_file_annotation(True)
  #         hyp = RevAI(hyp_path+hyp_file).get_file_annotation()
  #         error = simpleder.DER(ref, hyp)
  #         print("DER={:.3f}".format(error))
  
  reference = Annotation()
  reference[Segment(0, 10)] = 'A'
  reference[Segment(12, 20)] = 'B'
  reference[Segment(24, 27)] = 'A'
  reference[Segment(30, 40)] = 'C'
  
  hypothesis = Annotation()
  hypothesis[Segment(2, 13)] = 'a'
  hypothesis[Segment(13, 14)] = 'd'
  hypothesis[Segment(14, 20)] = 'b'
  hypothesis[Segment(22, 38)] = 'c'
  hypothesis[Segment(38, 40)] = 'd'
  
  metric = DiarizationErrorRate()
  metric(reference, hypothesis)