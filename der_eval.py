import simpleder
from TranscriptProcess import *
import os
from pyannote.core import Segment, Timeline, Annotation
from pyannote.metrics.diarization import DiarizationErrorRate


def precision(ref, hyp):
    """
    Presicion = (correctly annotated length in hyp) / (total length in hyp)

    Args:
        ref [(spk, start, end)]: ground truth annotation containning spk_id and start/end time of an utterance
        hyp [(spk, start, end, token)]: predicted annotation of each token
    """
    spk_map = {'A': 0, 'B': 1}  # need a method to determine this
    correct_count = 0
    for segment in ref:
        start, end = segment[1], segment[2]
        for token in hyp:
            # if compare_with_window(start, end, token[1], token[2]):
            # if start < token[1] and end > token[2]:
            if compare_with_window(start, end, token[1], token[2]):  # TODO: Some tokens might be counted twice.
                if spk_map[segment[0]] == token[0]:
                    correct_count += 1
            else:
                if abs(start - token[1]) < 0.5 or abs(end - token[2]) < 0.5:
                    print(token)
                    print(segment)

    return correct_count / len(hyp)


def compare_with_window(start, end, hyp_start, hyp_end):
    window = 0.5
    if start < hyp_start and end > hyp_end:
        return True
    elif hyp_end > start and abs(start - hyp_start) < window:  # at the beginning of a segement
        return True
    elif end > hyp_start and abs(end - hyp_end) < window:  # at the end of a segement
        return True
    elif abs(start - hyp_start) < window and hyp_end < start:  # missed at the beginning
        return True
    elif abs(end - hyp_end) < window and hyp_start > end:  # missed at the end
        return True
    return False


def simple_der():
    """Calculate the audio based der score using simpleder, which doesn't allow overlapping
       for the entire directory

    """
    ref_path = "CallHome_eval/transcripts/"
    hyp_dir_path = "CallHome_eval/amazon/"
    # hyp_dir_path = "CallHome_eval/rev/"
    ref_files = os.listdir(ref_path)
    hyp_files = os.listdir(hyp_dir_path)
    score = {}
    for file in ref_files:
        name = file[:file.find(".")]
        for hyp_file in hyp_files:
            if name in hyp_file:
                ref = CallHome(ref_path + file).get_file_annotation(start_with_zero=True, allow_overlap=False)
                # hyp = RevAI(hyp_path+hyp_file).get_file_annotation()
                hyp = Amazon(hyp_path + hyp_file).get_file_annotation()
                error = simpleder.DER(ref, hyp)
                score[name] = error
                print(name, "DER={:.3f}".format(error))
    print(sum(score.values()) / len(score.keys()))


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


if __name__ == "__main__":
    name1 = "CallHome_eval/transcripts/4093.cha"
    name2 = "CallHome_eval/rev/4093_cut.json"
    ref_path = "CallHome_eval/transcripts/"
    # hyp_path = "CallHome_eval/rev/"
    hyp_path = "CallHome_eval/amazon/"

    ref_files = os.listdir(ref_path)
    hyp_files = os.listdir(hyp_path)

    call_4074 = CallHome("CallHome_eval/transcripts/4074.cha")
    rev_4074 = RevAI("CallHome_eval/rev/4074_cut.json")
    ref = call_4074.get_file_annotation(with_utterances=True)
    hyp = rev_4074.get_spk_time_token()
    print(precision(ref, hyp))

    # print(call_4074.get_file_annotation(with_utterances=True))
