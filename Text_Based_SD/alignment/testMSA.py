from MSA import *
from NeedlemanWunsch3D import *
from Text_Based_SD.data.TranscriptProcess import CallHome, Amazon, RevAI

file_code = "4074_short_test"


def threeDimension(file_code):
  seq1 = [token.value for token in CallHome(f"../data/CallHome_eval/transcripts/{file_code}.cha").get_token_list() if
          token.spk_id == 'A']
  seq2 = [token.value for token in CallHome(f"../data/CallHome_eval/transcripts/{file_code}.cha").get_token_list() if
          token.spk_id == 'B']
  target = [token.value for token in
            RevAI(f"../data/CallHome_eval/rev/txt/{file_code}_cut.txt", istxt=True).get_token_list()]
  align1, align2, align3, align2_to_align1, align3_to_align1 = backtrack(target, seq1, seq2,
                                                                        get_scoring_matrix_3d(target, seq1, seq2,
                                                                                              file_code), file_code)
  return align1, align2, align3, align2_to_align1, align3_to_align1


def multiDimension(file_code):
  seq1 = [token.value for token in CallHome(f"../data/CallHome_eval/transcripts/{file_code}.cha").get_token_list() if
          token.spk_id == 'A']
  seq2 = [token.value for token in CallHome(f"../data/CallHome_eval/transcripts/{file_code}.cha").get_token_list() if
          token.spk_id == 'B']
  target = [token.value for token in
            RevAI(f"../data/CallHome_eval/rev/txt/{file_code}_cut.txt", istxt=True).get_token_list()]
  
  seqList = [seq1, seq2]
  
  result_list = multi_sequence_alignment(target, seqList)
  return result_list


align1, align2, align3, align2_to_align1, align3_to_align1 = threeDimension(file_code=file_code)
print( align2)
print("Start MSA\n")
result_list = multiDimension(file_code=file_code)

with open(f"{file_code}_MSA_test.csv", 'w') as file:
    output = csv.writer(file)
    output.writerows([align2, align3, align1, result_list[1], result_list[2], result_list[0]])