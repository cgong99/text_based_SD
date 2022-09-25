import simpleder
from TranscriptProcess import *


if __name__ == "__main__":
  name1 = "CallHome_eval/transcripts/4093.cha"
  name2 = "CallHome_eval/rev/4093_cut.json"
  ref = CallHome(name1).get_file_annotation(True)
  hyp = RevAI(name2).get_annotation()
  
  error = simpleder.DER(ref, hyp)
  print("DER={:.3f}".format(error))

  