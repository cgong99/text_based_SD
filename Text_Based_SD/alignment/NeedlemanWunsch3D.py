import csv
import math
from multiprocessing import Pool
import timeit

import numpy as np
from numba import jit
from numpy import ndarray

from NeedlemanWunsch import edit_distance, compare
from Text_Based_SD.data.TranscriptProcess import CallHome, Amazon, RevAI


@jit(nopython=True)
def compare_3d(target_token, token1, token2) -> int:
    mode = 0
    if target_token == '-':
        mode += 4
    if token1 == '-':
        mode += 2
    if token2 == '-':
        mode += 1

    if mode == 1:
        if edit_distance(target_token, token1) < 2:
            return 3
        else:
            return -1
    elif mode == 2:
        if edit_distance(target_token, token1) < 2:
            return 3
        else:
            return -1
    elif mode == 3 or mode == 5 or mode == 6:
        return 1
    elif mode == 0 or mode == 4:
        return -1

    # mode_to_score = {
    #     0: -1,
    #     1: 3 if edit_distance(target_token, token1) < 2 else -1,
    #     2: 3 if edit_distance(target_token, token2) < 2 else -1,
    #     3: 1,
    #     4: -1,
    #     5: 1,
    #     6: 1
    # }
    # return mode_to_score[mode]


@jit(nopython=True)
def get_scoring_matrix_3d(target_seq: list[str], seq1: list[str], seq2: list[str], file_code: str) -> ndarray:
    gap = -1
    count = 0
    progress = 0
    parameter_number = (len(target_seq) + 1) * (len(seq1) + 1) * (len(seq2) + 1)
    print(f"length of three sequence for {file_code}: {len(target_seq)}, {len(seq1)}, {len(seq2)}")
    print(f"total number of parameters for {file_code}: {parameter_number}")
    score = np.zeros((len(target_seq) + 1, len(seq1) + 1, len(seq2) + 1), dtype="int32")
    # initial three edges and three surfaces
    for i in range(0, len(target_seq) + 1):
        score[i][0][0] = gap * i
    for j in range(0, len(seq1) + 1):
        score[0][j][0] = gap * j
    for k in range(0, len(seq2) + 1):
        score[0][0][k] = gap * k
    for i in range(1, len(target_seq) + 1):
        for j in range(1, len(seq1) + 1):
            score[i][j][0] = max(score[i - 1][j - 1][0] + compare(target_seq[i - 1], seq1[j - 1]),
                                 score[i - 1][j][0] + gap, score[i][j - 1][0] + gap)
    for i in range(1, len(target_seq) + 1):
        for k in range(1, len(seq2) + 1):
            score[i][0][k] = max(score[i - 1][0][k - 1] + compare(target_seq[i - 1], seq2[k - 1]),
                                 score[i - 1][0][k] + gap, score[i][0][k - 1] + gap)
    for j in range(1, len(seq1) + 1):
        for k in range(1, len(seq2) + 1):
            score[0][j][k] = max(score[0][j - 1][k - 1] + compare(seq1[j - 1], seq2[k - 1]),
                                 score[0][j - 1][k] + gap, score[0][j][k - 1] + gap)

    for i in range(1, len(target_seq) + 1):
        for j in range(1, len(seq1) + 1):
            for k in range(1, len(seq2) + 1):
                if count % int(parameter_number / 100) == 0:
                    print(f"transcript {file_code}: matrix calculation progress {progress}%, i={i}, j={j}, k={k}")
                    progress += 1
                spec1 = score[i][j - 1][k - 1] + compare_3d('-', seq1[j - 1], seq2[k - 1])
                spec2 = score[i - 1][j][k - 1] + compare_3d(target_seq[i - 1], '-', seq2[k - 1])
                spec3 = score[i - 1][j - 1][k] + compare_3d(target_seq[i - 1], seq1[j - 1], '-')
                spec4 = score[i - 1][j][k] + compare_3d(target_seq[i - 1], '-', '-')
                spec5 = score[i - 1][j][k - 1] + compare_3d('-', seq1[j - 1], '-')
                spec6 = score[i][j][k - 1] + compare_3d('-', '-', seq2[k - 1])
                spec7 = score[i - 1][j - 1][k - 1] + compare_3d(target_seq[i - 1], seq1[j - 1], seq2[k - 1])
                score[i][j][k] = max(spec1, spec2, spec3, spec4, spec5, spec6, spec7)
                count += 1
    print(f"transcript {file_code}: matrix calculation complete!")
    return score


@jit(nopython=True)
def backtrack(target_seq, seq1, seq2, matrix, file_code: str):
    gap = -1
    i = len(target_seq)
    j = len(seq1)
    k = len(seq2)
    align1, align2, align3 = [], [], []
    align2_to_align1 = np.zeros(j + 1, dtype="int32")
    align3_to_align1 = np.zeros(k + 1, dtype="int32")
    progress = 100
    parameter_number = (len(target_seq)) * (len(seq1)) * (len(seq2))
    while i > 0 and j > 0 and k > 0:
        if math.floor(i * j * k * 100 / parameter_number) < progress:
            print(f"transcript {file_code}: backtrack progress {progress}%, i={i}, j={j}, k={k}")
            progress -= 1
        xi = target_seq[i - 1]
        yj = seq1[j - 1]
        zk = seq2[k - 1]
        if matrix[i, j, k] == compare_3d('-', yj, zk) + matrix[i, j - 1, k - 1]:
            align1.append('-')
            align2.append(yj)
            align2_to_align1[j] = -1
            align3.append(zk)
            align3_to_align1[k] = -1
            j -= 1
            k -= 1
        elif matrix[i, j, k] == compare_3d(xi, '-', zk) + matrix[i - 1, j, k - 1]:
            align1.append(xi)
            align2.append('-')
            align3.append(zk)
            align3_to_align1[k] = i
            i -= 1
            k -= 1
        elif matrix[i, j, k] == compare_3d(xi, yj, '-') + matrix[i - 1, j - 1, k]:
            align1.append(xi)
            align2.append(yj)
            align2_to_align1[j] = i
            align3.append('-')
            i -= 1
            j -= 1
        elif matrix[i, j, k] == compare_3d(xi, '-', '-') + matrix[i - 1, j, k]:
            align1.append(xi)
            align2.append('-')
            align3.append('-')
            i -= 1
        elif matrix[i, j, k] == compare_3d('-', yj, '-') + matrix[i, j - 1, k]:
            align1.append('-')
            align2.append(yj)
            align2_to_align1[j] = -1
            align3.append('-')
            j -= 1
        elif matrix[i, j, k] == compare_3d('-', '-', zk) + matrix[i, j, k - 1]:
            align1.append('-')
            align2.append('-')
            align3.append(zk)
            align3_to_align1[k] = -1
            k -= 1
        elif matrix[i, j, k] == compare_3d(xi, yj, zk) + matrix[i - 1, j - 1, k - 1]:
            align1.append(xi)
            align2.append(yj)
            align2_to_align1[j] = i
            align3.append(zk)
            align3_to_align1[k] = i
            i -= 1
            j -= 1
            k -= 1

    # one of the dimension is 0, on a surface now
    while i > 0 and j > 0:
        if math.floor(i * j * k * 100 / parameter_number) < progress:
            print(f"transcript {file_code}: backtrack progress {progress}%, i={i}, j={j}, k={k}")
            progress -= 1
        xi = target_seq[i - 1]
        yj = seq1[j - 1]
        if matrix[i, j, k] == compare(xi, yj) + matrix[i - 1, j - 1, k]:  # 2d using NeedlemanWunsch's compare
            align1.append(xi)
            align2.append(yj)
            align2_to_align1[j] = i
            align3.append('-')
            i -= 1
            j -= 1
        elif matrix[i, j, k] == matrix[i - 1, j, k] + gap:  # self.gap computes 2d pairwise alignment
            align1.append(xi)
            align2.append('-')
            align3.append('-')
            i -= 1
        elif matrix[i, j, k] == matrix[i, j - 1, k] + gap:
            align1.append('-')
            align2.append(yj)
            align2_to_align1[j] = -1
            align3.append('-')
            j -= 1

    while i > 0 and k > 0:
        if math.floor(i * j * k * 100 / parameter_number) < progress:
            print(f"transcript {file_code}: backtrack progress {progress}%, i={i}, j={j}, k={k}")
            progress -= 1
        xi = target_seq[i - 1]
        zk = seq2[k - 1]
        if matrix[i, j, k] == compare(xi, zk) + matrix[i - 1, j, k - 1]:  # 2d using NeedlemanWunsch's compare
            align1.append(xi)
            align2.append('-')
            align3.append(zk)
            align3_to_align1[k] = i
            i -= 1
            k -= 1
        elif matrix[i, j, k] == matrix[i - 1, j, k] + gap:
            align1.append(xi)
            align2.append('-')
            align3.append('-')
            i -= 1
        elif matrix[i, j, k] == matrix[i, j, k - 1] + gap:
            align1.append('-')
            align2.append('-')
            align3.append(zk)
            align3_to_align1[k] = -1
            k -= 1

    while j > 0 and k > 0:
        if math.floor(i * j * k * 100 / parameter_number) < progress:
            print(f"transcript {file_code}: backtrack progress {progress}%, i={i}, j={j}, k={k}")
            progress -= 1
        yj = seq1[j - 1]
        zk = seq2[k - 1]
        if matrix[i, j, k] == compare(yj, zk) + matrix[i, j - 1, k - 1]:  # 2d using NeedlemanWunsch's compare
            align1.append('-')
            align2.append(yj)
            align2_to_align1[j] = -1
            align3.append(zk)
            align3_to_align1[k] = -1
            j -= 1
            k -= 1
        elif matrix[i, j, k] == matrix[i, j - 1, k] + gap:
            align1.append('-')
            align2.append(yj)
            align2_to_align1[j] = -1
            align3.append('-')
            j -= 1
        elif matrix[i, j, k] == matrix[i, j, k - 1] + gap:
            align1.append('-')
            align2.append('-')
            align3.append(zk)
            align3_to_align1[k] = -1
            k -= 1

    while i > 0:
        if math.floor(i * j * k * 100 / parameter_number) < progress:
            print(f"transcript {file_code}: backtrack progress {progress}%, i={i}, j={j}, k={k}")
            progress -= 1
        xi = target_seq[i - 1]
        align1.append(xi)
        align2.append('-')
        align3.append('-')
        i -= 1

    while j > 0:
        if math.floor(i * j * k * 100 / parameter_number) < progress:
            print(f"transcript {file_code}: backtrack progress {progress}%, i={i}, j={j}, k={k}")
            progress -= 1
        yj = seq1[j - 1]
        align1.append('-')
        align2.append(yj)
        align2_to_align1[j] = -1
        align3.append('-')
        j -= 1

    while k > 0:
        if math.floor(i * j * k * 100 / parameter_number) < progress:
            print(f"transcript {file_code}: backtrack progress {progress}%, i={i}, j={j}, k={k}")
            progress -= 1
        zk = seq2[k - 1]
        align1.append('-')
        align2.append('-')
        align3.append(zk)
        align3_to_align1[k] = -1
        k -= 1
    print(f"transcript {file_code}: backtrack complete!")
    return align1[::-1], align2[::-1], align3[::-1], align2_to_align1, align3_to_align1


def write_csv(file_code: str):
    seq1 = [token.value for token in CallHome(f"../data/CallHome_eval/transcripts/{file_code}.cha").get_token_list() if
            token.spk_id == 'A']
    seq2 = [token.value for token in CallHome(f"../data/CallHome_eval/transcripts/{file_code}.cha").get_token_list() if
            token.spk_id == 'B']
    # seq1 = [token.value for token in Amazon(f"../data/CallHome_eval/amazon/{file_code}.json").get_token_list() if
    #         token.spk_id == "spk_0"]
    # seq2 = [token.value for token in Amazon(f"../data/CallHome_eval/amazon/{file_code}.json").get_token_list() if
    #         token.spk_id == "spk_1"]
    # target = [token.value for token in CallHome(f"../data/CallHome_eval/transcripts/{file_code}.cha").get_token_list()]

    # target = [token.value for token in Amazon(f"../data/CallHome_eval/amazon/{file_code}.json").get_token_list()]
    target = [token.value for token in RevAI(f"../data/CallHome_eval/rev/{file_code}_cut.json").get_token_list()]
    align1, align2, align3, align2_to_align1, align3_to_align1 = backtrack(target, seq1, seq2,
                                                                           get_scoring_matrix_3d(target, seq1, seq2,
                                                                                                 file_code), file_code)
    with open(f"../alignment/ResultRevAI/Result3D/{file_code}_result_revai.csv", 'w') as file:
        output = csv.writer(file)
        output.writerows([align2, align3, align1, align2_to_align1, align3_to_align1])
    print(f"{file_code} has been written.\n")


def test_performance():
    file_code = "4074"
    seq1 = [token.value for token in CallHome(f"../data/CallHome_eval/transcripts/{file_code}.cha").get_token_list() if
            token.spk_id == 'A']
    seq2 = [token.value for token in CallHome(f"../data/CallHome_eval/transcripts/{file_code}.cha").get_token_list() if
            token.spk_id == 'B']
    target = [token.value for token in RevAI(f"../data/CallHome_eval/rev/{file_code}_cut.json").get_token_list()]
    align1, align2, align3, align2_to_align1, align3_to_align1 = backtrack(target, seq1, seq2,
                                                                           get_scoring_matrix_3d(target, seq1, seq2,
                                                                                                 file_code), file_code)
    with open(f"../alignment/ResultRevAI/Result3D/{file_code}_test_performance_revai.csv", 'w') as file:
        output = csv.writer(file)
        output.writerows([align2, align3, align1, align2_to_align1, align3_to_align1])
    print("test complete!")


if __name__ == "__main__":
    print(timeit.Timer(test_performance).timeit(number=1))
    # with Pool(1) as pool:
    #     pool.map(write_csv, ["4074", "4093", "4247", "4315", "4325", "4335", "4571", "4595", "4660", "4290"])
    # pool.map(write_csv, ["4315", "4325", "4335", "4571", "4595", "4660", "4290"])
