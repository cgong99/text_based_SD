from NeedlemanWunsch3D import *
from Text_Based_SD.data.TranscriptProcess import Amazon, RevAI, CallHome
from Text_Based_SD.data.TranscriptProcess import *

def test_single_segments():
  gt = CallHome("../data/CallHome_eval/transcripts/4074.cha").get_token_list()
  target = Amazon("../data/CallHome_eval/amazon/4074.json").get_token_list()
  gt_segs, target_segs = segment_token_lists(gt, target)
  segment_count = len(gt_segs)
  gt_tokens = gt_segs[1]
  target_tokens = target_segs[1]
  seq1 = [token.value for token in gt_tokens if token.spk_id == 'A']
  seq2 = [token.value for token in gt_tokens if token.spk_id == 'B']
  target_seq = [token.value for token in target_tokens]

  # print(len(seq1))
  # print(len(target))
  # print(seq1)
  # print(seq2)
  # print(target_seq)
  # align = MultiSeqAlign(target_seq, seq1, seq2)
  # align.compute_matrix()
  # target_align, seq1_align, seq2_align = align.backtrack()

  align1, align2, align3, align2_to_align1, align3_to_align1= backtrack(target_seq, seq1, seq2, get_scoring_matrix_3d(target_seq, seq1, seq2, "test"), "test")
  with open("4074_single_segment_test2.csv", 'w') as file:
    output = csv.writer(file)
    # output.writerows([seq1_align, seq2_align, target_align])
    output.writerows([align2, align3, align1, align2_to_align1, align3_to_align1])
  
def test_segments():
  gt = CallHome("../data/CallHome_eval/transcripts/4074.cha").get_token_list()
  target = Amazon("../data/CallHome_eval/amazon/4074.json").get_token_list()
  gt_segs, target_segs = segment_token_lists(gt, target)
  segment_count = len(gt_segs)
  print("segments: ", segment_count)
  all_target, all_align2, all_align3, all_2to1, all_3to1 = [], [], [], [], []
  
  prev_length = 0
  for i in range(segment_count):
    print("\ncomputing segment: ", i+1)
    seq1 = [token.value for token in gt_segs[i] if token.spk_id == 'A']
    seq2 = [token.value for token in gt_segs[i] if token.spk_id == 'B']
    target_seq = [token.value for token in target_segs[i]]
    # print(target_seq)
    align1, align2, align3, align2_to_align1, align3_to_align1= backtrack(target_seq, seq1, seq2, get_scoring_matrix_3d(target_seq, seq1, seq2, "test"), "test")
    align2_to_align1 = update_align(align2_to_align1, prev_length)
    align3_to_align1 = update_align(align3_to_align1, prev_length)
    all_align2.extend(align2)
    all_align3.extend(align3)
    all_target.extend(align1)
    all_2to1.extend(align2_to_align1)
    all_3to1.extend(align3_to_align1)
    prev_length += len(target_seq)
  
  with open("4074_result_amazon.csv", 'w') as file:
        output = csv.writer(file)
        output.writerows([all_align2, all_align3, all_target, all_2to1, all_3to1])

def update_align(align, prev_length):
  res = []
  for a in align:
    if a != -1:
      a += prev_length
    res.append(a)
  return res

if __name__ == "__main__":
  # test_single_segments()
  test_segments()
