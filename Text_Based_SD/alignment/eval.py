import csv
import pandas as pd
import numpy as np
from Text_Based_SD.data.TranscriptProcess import *

rev_spk = [0, 1]
amazon_spk = ["spk_0", "spk_1"]


def removeNan(row):
    res = []
    for i in row:
        if float(i) >= -1 and float(i) != 0:
            res.append(float(i))
    return res


def mapSpeaker(gt_tokens: list[Token], hyp_tokens: list[Token], type):
    gt_spk1 = gt_tokens[0].spk_id
    gt_spk2 = None
    for token in gt_tokens:
        if token.spk_id != gt_spk1:
            gt_spk2 = token.spk_id
            break
    hyp_spk1 = hyp_tokens[0].spk_id
    hyp_spk2 = None
    for token in hyp_tokens:
        if token.spk_id != hyp_spk1:
            hyp_spk2 = token.spk_id
            break
    if not hyp_spk2:
        if type == "Rev":
            if hyp_spk1 == rev_spk[0]:
                hyp_spk2 = rev_spk[1]
            else:
                hyp_spk2 = rev_spk[0]
    # tmp_map = {gt_spk1: hyp_spk1, gt_spk2: hyp_spk2}
    spk1_count, spk2_count = 0, 0
    for i in range(len(gt_tokens)):
        tk = gt_tokens[i]
        if tk.spk_id == gt_spk1:
            spk1_count += 1
        else:
            spk2_count += 1
    hyp_spk1_count, hyp_spk2_count = 0, 0
    for i in range(len(hyp_tokens)):
        tk = hyp_tokens[i]
        if tk.spk_id == hyp_spk1:
            hyp_spk1_count += 1
        else:
            hyp_spk2_count += 1
    
    if (spk1_count > spk2_count and hyp_spk1_count > hyp_spk2_count)or(spk1_count< spk2_count and hyp_spk1_count < hyp_spk2_count):
        return {gt_spk1: hyp_spk1, gt_spk2: hyp_spk2}
    else:
        return {gt_spk1: hyp_spk2, gt_spk2: hyp_spk1}



def get_speaker_list(token_list):
    spk1 = token_list[0].spk_id
    for token in token_list:
        if token.spk_id != spk1:
            spk2 = token.spk_id
            return (spk1, spk2)


class Eval_3d:
    def __init__(self, file_code, type, path=None) -> None:
        self.file_code = file_code
        self.type = type
        if type == "Amazon":
            self.resultPath = f"ResultAmazon/Result3D/{self.file_code}_result_amazon.csv"
            self.hyp_raw_file = Amazon(f"../data/CallHome_eval/amazon/{self.file_code}.json")
            self.hyp_tokens = self.hyp_raw_file.get_token_list()
            
        if type == "Rev":
            self.resultPath = f"ResultRevAI/Result3D/{self.file_code}_result_revai.csv"
            self.hyp_raw_file = RevAI(f"../data/CallHome_eval/rev/{self.file_code}_cut.json")
            self.hyp_tokens = self.hyp_raw_file.get_token_list()
        if path:
            self.resultPath = path  # if specified result csv path

        self.spk1_align, self.spk2_align = self.read3dResult()
        self.gt_file =  CallHome(f"../data/CallHome_eval/transcripts/{self.file_code}.cha")
        self.gt_tokens = self.gt_file.get_token_list()
        self.gt_spk_ids = get_speaker_list(self.gt_tokens)
        self.hyp_spk_ids = get_speaker_list(self.hyp_tokens)
        self.spk_map = mapSpeaker(self.gt_tokens, self.hyp_tokens, type)

    def read3dResult(self):
        df = pd.read_csv(self.resultPath, header=None).fillna(-10)
        spk1_align_raw = df.values[3, 1:]
        spk2_align_raw = df.values[4, 1:]
        spk1_align = removeNan(spk1_align_raw)
        spk2_align = removeNan(spk2_align_raw)
        return spk1_align, spk2_align

    def getHyp2SpkAlignment(self):
        """base on two speakers alignment, generate hyp_to_spk alignment

        Returns:
            [a-1, a-2, b-1, b-2,...]
        """
        hyp_length = len(self.hyp_tokens)
        hyp_align = ["-" for i in range(hyp_length)]
        for i, align in enumerate(self.spk1_align):
            align = int(align-1)
            if align > 0:
                # hyp_align[align] = str(self.gt_spk_ids[0]) + "-" + str(i)
                hyp_align[align] =  "A-" + str(i)
            
        for i, align in enumerate(self.spk2_align):
            align = int(align-1)
            if align > 0:
                # hyp_align[align] = str(self.gt_spk_ids[1]) + "-" + str(i)
                hyp_align[align] =  "B-" + str(i)
            
        return hyp_align
        
    
    def calculate(self):
        error_count, correct_count, gap = 0, 0, 0
        spk1_error, spk2_error = [], []
        for i in range(len(self.spk1_align)):
            index = int(self.spk1_align[i]) - 1
            if index < 0:
                gap += 1
                continue
            if self.hyp_tokens[index].spk_id == self.spk_map[self.gt_spk_ids[0]]:  # gt first speaker
                correct_count += 1
            else:
                error_count += 1
                spk1_error.append(i)

        for i in range(len(self.spk2_align)):
            index = int(self.spk2_align[i]) - 1
            if index < 0:
                gap += 1
                continue
            if self.hyp_tokens[index].spk_id == self.spk_map[self.gt_spk_ids[1]]:
                correct_count += 1
            else:
                error_count += 1
                spk2_error.append(i)

        correct = max(correct_count, error_count)
        error = min(correct_count, error_count)
        recall = correct / len(self.gt_tokens)
        print("Error: ", error)
        print("Correct: ", correct)
        print("Gap: ", gap)
        print("Token Count: ", len(self.gt_tokens))
        print("Correct Rate: ", correct / len(self.gt_tokens))
        print("Error Rate: ", error / len(self.gt_tokens))
        print("Recall: ", correct / len(self.gt_tokens))
        return error, correct, gap, recall

    def precision(
            self):  # having two speaker alignment (ground truth to target), find the target to gt alignment and count precision
        token_num = len(self.hyp_tokens)
        hyp_align = [-1 for i in range(token_num)]
        for i, index in enumerate(self.spk1_align):  # first gt speaker
            index = int(index) - 1
            if hyp_align[index] != 1 and self.spk_map[self.gt_spk_ids[0]] == self.hyp_tokens[index].spk_id:
                hyp_align[index] = 1
            else:
                hyp_align[index] = 0

        for i, index in enumerate(self.spk2_align):
            index = int(index) - 1
            if hyp_align[index] != 1 and self.spk_map[self.gt_spk_ids[1]] == self.hyp_tokens[index].spk_id:
                hyp_align[index] = 1
            else:
                hyp_align[index] = 0

        correct, error, gap = 0, 0, 0
        for align in hyp_align:
            if align == 1:
                correct += 1
            elif align == 0:
                error += 1
            elif align == -1:
                gap += 1
            else:
                print("####### error unexpected align result ")

        print(correct)
        print(error)
        print(gap)
        correct = max(correct, error)
        error = min(correct, error)

        return correct / token_num, correct, error, gap


class Eval_2d():
    def __init__(self, file_code, type) -> None:
        self.file_code = file_code
        self.gt_tokens = CallHome(f"../data/CallHome_eval/transcripts/{self.file_code}.cha").get_token_list()
        if type == "Amazon":
            self.resultPath = f"ResultAmazon/Result2DCombined/{file_code}_result_amazon.csv"
            self.hyp_tokens = Amazon(f"../data/CallHome_eval/amazon/{self.file_code}.json").get_token_list()
        elif type == "Rev":
            self.resultPath = f"ResultRevAI/Result2DCombined/{file_code}_result_revai.csv"
            self.hyp_tokens = RevAI(f"../data/CallHome_eval/rev/{self.file_code}_cut.json").get_token_list()
        self.gt_to_hyp, self.hyp_to_gt = self.read2dResult()
        self.spk_map = mapSpeaker(self.gt_tokens, self.hyp_tokens, type)

    def read2dResult(self):
        df = pd.read_csv(self.resultPath, header=None).fillna(-10)
        gt_align_hyp_raw = df.values[2, 1:]
        hyp_align_gt_raw = df.values[3, 1:]
        gt_align_hyp = removeNan(gt_align_hyp_raw)
        hyp_align_gt = removeNan(hyp_align_gt_raw)
        return gt_align_hyp, hyp_align_gt

    def calculate(self):
        correct_count, error_count, gap = 0, 0, 0
        for i in range(len(self.gt_to_hyp)):
            index = i - 1
            hyp_index = int(self.gt_to_hyp[i]) - 1
            gt_spk = self.gt_tokens[index].spk_id
            hyp_spk = self.hyp_tokens[hyp_index].spk_id
            if hyp_index < 0:
                gap += 1
                continue
            if hyp_spk == self.spk_map[gt_spk]:
                correct_count += 1
            else:
                error_count += 1

        print("Error: ", error_count)
        print("Correct: ", correct_count)
        print("Gap: ", gap)
        print("Token Count: ", len(self.gt_tokens))
        print("Correct Rate: ", correct_count / len(self.gt_tokens))
        print("Error Rate: ", error_count / len(self.gt_tokens))
        print("Recall: ", correct_count / len(self.gt_tokens))
        return error_count, correct_count, gap


def precision_by_time(gt_segments, gt_tokens, hyp_tokens, type):
    spk_map = mapSpeaker(gt_tokens, hyp_tokens, type)
    correct_count = 0
    for segment in gt_segments:
        start, end = segment[1], segment[2]
        for token in hyp_tokens:
            # if compare_with_window(start, end, token[1], token[2]):
            # if start < token[1] and end > token[2]:
            if compare_with_window(start, end, token.start, token.end):  # TODO: Some tokens might be counted twice.
                if spk_map[segment[0]] == token.spk_id:
                    correct_count += 1
            else:
                if abs(start - token.start) < 0.1 or abs(end - token.end) < 0.1:
                    print(token)
                    print(segment)
    print(len(hyp_tokens))
    print(correct_count)
    return correct_count / len(hyp_tokens)


def compare_with_window(start, end, hyp_start, hyp_end):
    window = 0.0
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


def output_3d_eval_csv():
    header = ["file", "error_rate", "correct_rate (recall)", "Gap_rate", "error_count", "correct_count", "Gap",
              "total_token", "recall", "precision", "F1", "Hyp_correct", "Hyp_error", "Hyp_gap", "hyp_token_len"]
    pool = ["4074", "4315", "4093", "4247", "4325", "4335", "4571", "4595"]  # deleted 4290 4660
    # with open("ResultAmazon/Amazon_3D_eval.csv", 'w') as file:
    with open("ResultRevAI/Rev_3D_eval.csv", 'w') as file:
        output = csv.writer(file)
        output.writerow(header)
        for file_code in pool:
            print("======= ", file_code)
            eval = Eval_3d(file_code=file_code, type="Rev")
            # eval = Eval_3d(file_code=file_code, type="Amazon")
            error, correct, gap, recall = eval.calculate()
            precision, hyp_correct, hyp_error, hyp_gap = eval.precision()
            token_len = len(eval.gt_tokens)
            hyp_token_len = len(eval.hyp_tokens)
            F1 = 2 * precision * recall / (precision + recall)
            row = [file_code, error / token_len, correct / token_len, gap / token_len, error, correct, gap, token_len,
                   recall, precision, F1, hyp_correct, hyp_error, hyp_gap, hyp_token_len]
            output.writerow(row)
 

def output_2d_eval_csv():
    header = ["file", "error_rate", "correct_rate (recall)", "Gap_rate", "error_count", "correct_count", "Gap",
              "total_token"]
    pool = ["4074", "4315", "4093", "4247", "4325", "4335", "4571", "4595", "4660"]
    # with open("ResultAmazon/Amazon_2D_eval.csv", 'w') as file:
    with open("ResultRevAI/Rev_2D_eval.csv", 'w') as file:
        output = csv.writer(file)
        output.writerow(header)
        for file_code in pool:
            print("======= ", file_code)
            eval = Eval_2d(file_code=file_code, type="Rev")
            error, correct, gap = eval.calculate()
            token_len = len(eval.gt_tokens)
            row = [file_code, error / token_len, correct / token_len, gap / token_len, error, correct, gap, token_len]
            output.writerow(row)


if __name__ == "__main__":
    file_code = 4074
    eval = Eval_3d(file_code=file_code, type="Rev", path=f"../alignment/ResultRevAI/Result3D/{file_code}_result_revai.csv")
    error, correct, gap, recall = eval.calculate()
    precision, correct, error, gap= eval.precision()
    print("precision: ", precision)
    print("F1: ",   2* precision * recall/ (precision + recall))

    # eval = Eval_3d(file_code=file_code, type="Rev")
    # eval.calculate()
    # eval.back_align()
    # print(eval.gt_spk_ids)
    # print(eval.hyp_spk_ids)
    # print(eval.spk_map)

    # hyp_tokens = Amazon(f"../data/CallHome_eval/amazon/4074.json").get_token_list()
    # # hyp_tokens = RevAI(f"../data/CallHome_eval/rev/4074_cut.json").get_token_list()
    # gt = CallHome(f"../data/CallHome_eval/transcripts/4074.cha")
    # gt_segments = gt.get_file_annotation(with_utterances=True)
    # gt_tokens = gt.get_token_list()
    # print(precision_by_time(gt_segments, gt_tokens, hyp_tokens, "Amazon"))

    # file_code = 4074
    # eval = Eval_2d(file_code=file_code)
    # eval.calculate()

    output_3d_eval_csv()

    # output_2d_eval_csv()
