import numpy as np
from numpy import ndarray
from typing import List, Tuple
from numba import jit
from NeedlemanWunsch import edit_distance, get_scoring_matrix, compare
from Text_Based_SD.data.TranscriptProcess import CallHome, Amazon
import csv
from multiprocessing import Pool


@jit(nopython=True)
def score(token1, token2, token3) -> int:
    """return compare score for three tokens

    Args:
        token1 (string):
        token2 (string):
        token3 (string):

  Args:
      token1 (string): from target sequence
      token2 (string): from speaker 1
      token3 (string): from speaker 2

  Returns:
      int: score
  """
    if token1 != '-' and token2 != '-' and token3 == '-' and edit_distance(token1, token2) < 2:
        return 2
    elif token1 != '-' and token3 != '-' and token2 == '-' and edit_distance(token1, token3) < 2:
        return 2
    if token1 != '-' and token2 != '-' and token3 != '-':
        return 0
    return -2


@jit(nopython=True)
def compare_3d(token1, token2, token3) -> int:
    if token1 != '-' and token2 != '-' and token3 == '-' and edit_distance(token1, token2) < 2:
        return 2
    elif token1 != '-' and token3 != '-' and token2 == '-' and edit_distance(token1, token3) < 2:
        return 2
    if token1 != '-' and token2 != '-' and token3 != '-':
        return 0
    return -2


@jit(nopython=True)
def get_scoring_matrix_3d(seq1: list[str], seq2: list[str], seq3: list[str], file_code: str) -> ndarray:
    gap = -1
    count = 0
    progress = 0
    parameter_number = (len(seq1) + 1) * (len(seq2) + 1) * (len(seq3) + 1)
    print(f"length of three sequence: {len(seq1)}, {len(seq2)}, {len(seq3)}")
    print(f"total number of parameters for {file_code}: {parameter_number}")
    score = np.zeros((len(seq1) + 1, len(seq2) + 1, len(seq3) + 1))
    for i in range(0, len(seq1) + 1):
        score[i][0][0] = gap * i
    for j in range(0, len(seq2) + 1):
        score[0][j][0] = gap * j
    for k in range(0, len(seq3) + 1):
        score[0][0][k] = gap * k
    for i in range(1, len(seq1) + 1):
        for j in range(1, len(seq2) + 1):
            for k in range(1, len(seq3) + 1):
                if count % int(parameter_number / 100) == 0:
                    print(f"transcript {file_code}: matrix calculation progress {progress}%, i={i}, j={j}, k={k}")
                    progress += 1
                spec1 = score[i][j - 1][k - 1] + compare_3d('-', seq2[j - 1], seq3[k - 1])
                spec2 = score[i - 1][j][k - 1] + compare_3d(seq1[i - 1], '-', seq3[k - 1])
                spec3 = score[i - 1][j - 1][k] + compare_3d(seq1[i - 1], seq2[j - 1], '-')
                spec4 = score[i - 1][j][k] + compare_3d(seq1[i - 1], '-', '-')
                spec5 = score[i - 1][j][k - 1] + compare_3d('-', seq2[j - 1], '-')
                spec6 = score[i][j][k - 1] + compare_3d('-', '-', seq3[k - 1])
                spec7 = score[i - 1][j - 1][k - 1] + compare_3d(seq1[i - 1], seq2[j - 1], seq3[k - 1])
                score[i][j][k] = max(spec1, spec2, spec3, spec4, spec5, spec6, spec7)
                count += 1
    return score


@jit(nopython=True)
def backtrack(seq1, seq2, seq3, matrix, file_code:str):
    gap = -1
    i = len(seq1)
    j = len(seq2)
    k = len(seq3)
    align1, align2, align3 = [], [], []
    align2_to_align1 = np.zeros(j + 1)
    align3_to_align1 = np.zeros(k + 1)
    progress = 0
    parameter_number = (len(seq1)) * (len(seq2)) * (len(seq3))
    while i > 0 and j > 0 and k > 0:
        if (i * j * k) % int(parameter_number / 100) == 0:
            print(f"transcript {file_code}: backtrack progress {progress}%, i={i}, j={j}, k={k}")
            progress += 1
        xi = seq1[i - 1]
        yj = seq2[j - 1]
        zk = seq3[k - 1]
        if matrix[i, j, k] == score('-', yj, zk) + matrix[i, j - 1, k - 1]:
            align1.append('-')
            align2.append(yj)
            align2_to_align1[j] = -1
            align3.append(zk)
            align3_to_align1[k] = -1
            j -= 1
            k -= 1
        elif matrix[i, j, k] == score(xi, '-', zk) + matrix[i - 1, j, k - 1]:
            align1.append(xi)
            align2.append('-')
            align3.append(zk)
            align3_to_align1[k] = i
            i -= 1
            k -= 1
        elif matrix[i, j, k] == score(xi, yj, '-') + matrix[i - 1, j - 1, k]:
            align1.append(xi)
            align2.append(yj)
            align2_to_align1[j] = i
            align3.append('-')
            i -= 1
            j -= 1
        elif matrix[i, j, k] == score(xi, '-', '-') + matrix[i - 1, j, k]:
            align1.append(xi)
            align2.append('-')
            align3.append('-')
            i -= 1
        elif matrix[i, j, k] == score('-', yj, '-') + matrix[i, j - 1, k]:
            align1.append('-')
            align2.append(yj)
            align2_to_align1[j] = -1
            align3.append('-')
            j -= 1
        elif matrix[i, j, k] == score('-', '-', zk) + matrix[i, j, k - 1]:
            align1.append('-')
            align2.append('-')
            align3.append(zk)
            align3_to_align1[k] = -1
            k -= 1
        elif matrix[i, j, k] == score(xi, yj, zk) + matrix[i - 1, j - 1, k - 1]:
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
        if (i * j * k) % int(parameter_number / 100) == 0:
            print(f"transcript {file_code}: backtrack progress {progress}%, i={i}, j={j}, k={k}")
            progress += 1
        xi = seq1[i - 1]
        yj = seq2[j - 1]
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
        if (i * j * k) % int(parameter_number / 100) == 0:
            print(f"transcript {file_code}: backtrack progress {progress}%, i={i}, j={j}, k={k}")
            progress += 1
        xi = seq1[i - 1]
        zk = seq3[k - 1]
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
        if (i * j * k) % int(parameter_number / 100) == 0:
            print(f"transcript {file_code}: backtrack progress {progress}%, i={i}, j={j}, k={k}")
            progress += 1
        yj = seq2[j - 1]
        zk = seq3[k - 1]
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
        if (i * j * k) % int(parameter_number / 100) == 0:
            print(f"transcript {file_code}: backtrack progress {progress}%, i={i}, j={j}, k={k}")
            progress += 1
        xi = seq1[i - 1]
        align1.append(xi)
        align2.append('-')
        align3.append('-')
        i -= 1

    while j > 0:
        if (i * j * k) % int(parameter_number / 100) == 0:
            print(f"transcript {file_code}: backtrack progress {progress}%, i={i}, j={j}, k={k}")
            progress += 1
        yj = seq2[j - 1]
        align1.append('-')
        align2.append(yj)
        align2_to_align1[j] = -1
        align3.append('-')
        j -= 1

    while k > 0:
        if (i * j * k) % int(parameter_number / 100) == 0:
            print(f"transcript {file_code}: backtrack progress {progress}%, i={i}, j={j}, k={k}")
            progress += 1
        zk = seq3[k - 1]
        align1.append('-')
        align2.append('-')
        align3.append(zk)
        align3_to_align1[k] = -1
        k -= 1
    print(f"transcript {file_code}: backtrack complete!")
    return align1[::-1], align2[::-1], align3[::-1], align2_to_align1, align3_to_align1


def write_csv_amazon(file_code: str):
    seq1 = [token.value for token in Amazon(f"../data/CallHome_eval/amazon/{file_code}.json").get_token_list() if
            token.spk_id == "spk_0"]
    seq2 = [token.value for token in Amazon(f"../data/CallHome_eval/amazon/{file_code}.json").get_token_list() if
            token.spk_id == "spk_1"]
    target = [token.value for token in CallHome(f"../data/CallHome_eval/transcripts/{file_code}.cha").get_token_list()]
    align1, align2, align3, align2_to_align1, align3_to_align1 = backtrack(target, seq1, seq2, get_scoring_matrix_3d(target, seq1, seq2, file_code), file_code)
    with open(f"{file_code}_result_amazon.csv", 'w') as file:
        output = csv.writer(file)
        output.writerows([align2, align3, align1, align2_to_align1, align3_to_align1])
    print(f"{file_code} has been written.\n")


if __name__ == "__main__":
    with Pool(1) as pool:
        # pool.map(write_csv_amazon, ["4074", "4093", "4247", "4315", "4325", "4335", "4571", "4595", "4660", "4290"])
        pool.map(write_csv_amazon, ["4315", "4325", "4335", "4571", "4595", "4660", "4290"])
