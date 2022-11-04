import csv
from multiprocessing import Pool
from typing import List, Tuple

import numpy as np
from numba import jit
from numpy import ndarray

from Text_Based_SD.data.TranscriptProcess import *


@jit(nopython=True)
def edit_distance(token1: str, token2: str) -> int:
    """
    Compute the Levenshtein distance between two string
    :param token1: token as string
    :param token2: token as string
    :return: Levenshtein distance in int
    """
    if token1 == token2:
        return 0
    matrix = np.zeros((len(token1) + 1, len(token2) + 1), dtype="int32")
    for i in range(1, len(token1) + 1):
        matrix[i][0] = i
    for j in range(1, len(token2) + 1):
        matrix[0][j] = j
    for i in range(1, len(token1) + 1):
        for j in range(1, len(token2) + 1):
            substitution = 0 if token1[i - 1] == token2[j - 1] else 1
            matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + substitution)
    return matrix[len(token1)][len(token2)]


@jit(nopython=True)
def compare(token1: str, token2: str, distance_bound: int = 2, match=2, mis_match=-1, gap=-1) -> int:
    """
    Compare two string and determine if they are match,
    if the levenshtein distance is below the distance bound then count as match
    :param token1: token as string
    :param token2: token as string
    :param distance_bound: bound of levenshtein distance
    :return: integer represent the score of match, mis-match, or a gap
    """
    if edit_distance(token1, token2) <= distance_bound:
        return match
    elif token1 == '-' or token2 == '-':
        return gap
    else:
        return mis_match


@jit(nopython=True)
def get_scoring_matrix(seq1: List[str], seq2: List[str], match=2, mis_match=-1, gap=-1) -> ndarray:
    """
    Compute the scoring matrix for Needleman-Wunsch algorithm
    :param seq1: list of words from the dialogue
    :param seq2: list of words from the dialogue
    :return: 2d numpy array as the scoring matrix
    """
    # compute scoring matrix
    score = np.zeros((len(seq1) + 1, len(seq2) + 1), dtype="int32")
    for i in range(0, len(seq1) + 1):
        score[i][0] = gap * i
    for j in range(0, len(seq2) + 1):
        score[0][j] = gap * j
    for i in range(1, len(seq1) + 1):
        for j in range(1, len(seq2) + 1):
            score[i][j] = max(score[i - 1][j - 1] + compare(seq1[i - 1], seq2[j - 1], match, mis_match, gap),
                              score[i - 1][j] + gap, score[i][j - 1] + gap)
    return score


@jit(nopython=True)
def backtrack(seq1: List[str], seq2: List[str], score: ndarray, match, mis_match, gap) -> Tuple[
    List[str], ndarray, List[str], ndarray]:
    """
    Backtrack according to the scoring matrix to get the alignment result
    :param seq1: list of words from the dialogue, same as get scoring matrix
    :param seq2: list of words from the dialogue, same as get scoring matrix
    :param score: the scoring matrix from get scoring matrix
    :return: a tuple containing the list of aligned two sequence and their mapping to each other result
    """
    i = len(seq1)
    j = len(seq2)
    align1 = []
    align1_to_align2 = np.zeros(i + 1, dtype="int32")
    align2 = []
    align2_to_align1 = np.zeros(j + 1, dtype="int32")
    while i > 0 and j > 0:
        if score[i][j] == score[i - 1][j - 1] + compare(seq1[i - 1], seq2[j - 1], match, mis_match, gap):
            align1.append(seq1[i - 1])
            align1_to_align2[i] = j
            align2.append(seq2[j - 1])
            align2_to_align1[j] = i
            i -= 1
            j -= 1
        elif score[i][j] == score[i - 1][j] + gap:
            align1.append(seq1[i - 1])
            align1_to_align2[i] = -1
            align2.append('-')
            i -= 1
        elif score[i][j] == score[i][j - 1] + gap:
            align1.append('-')
            align2.append(seq2[j - 1])
            align2_to_align1[j] = -1
            j -= 1

    while i > 0:
        align1.append(seq1[i - 1])
        align2.append('-')
        i -= 1
    while j > 0:
        align1.append('-')
        align2.append(seq2[j - 1])
        j -= 1
    align1 = align1[::-1]
    align2 = align2[::-1]

    return align1, align1_to_align2, align2, align2_to_align1


def needleman_wunsch(seq1: List[str], seq2: List[str], match=2, mis_match=-1, gap=-1) -> Tuple[
    List[str], ndarray, List[str], ndarray]:
    score = get_scoring_matrix(seq1, seq2, match, mis_match, gap)
    return backtrack(seq1, seq2, score, match, mis_match, gap)


def test_needleman_wunsch():
    seq1 = [token.value for token in RevAI("../data/CallHome_eval/rev/4074_cut.json").get_token_list() if
            token.spk_id == 0]
    seq2 = [token.value for token in RevAI("../data/CallHome_eval/rev/4074_cut.json").get_token_list() if
            token.spk_id == 1]
    seq3 = [token.value for token in CallHome("../data/CallHome_eval/transcripts/4074.cha").get_token_list()]
    align13, map13, align31, map31 = needleman_wunsch(seq1, seq3)
    align23, map23, align32, map32 = needleman_wunsch(seq2, seq3)
    print(align13)
    print(align31)
    print(align23)
    print(align32)


def write_csv_combined(file_code: str, match, mis_match, gap):
    seq1 = [token.value for token in CallHome(f"../data/CallHome_eval/transcripts/{file_code}.cha").get_token_list()]
    seq2 = [token.value for token in RevAI(f"../data/CallHome_eval/rev/{file_code}_cut.json").get_token_list()]
    align12, map12, align21, map21 = needleman_wunsch(seq1, seq2, match, mis_match, gap)
    with open(f"../alignment/ResultRevAI/Result2DCombined/{file_code}_result_revai.csv", 'w') as file:
        output = csv.writer(file)
        output.writerows([align12, align21, map12, map21])


def test_scoring(param_list: list[int]):
    file_code = "4074"
    seq1 = [token.value for token in CallHome(f"../data/CallHome_eval/transcripts/{file_code}.cha").get_token_list()]
    seq2 = [token.value for token in RevAI(f"../data/CallHome_eval/rev/{file_code}_cut.json").get_token_list()]
    align12, map12, align21, map21 = needleman_wunsch(seq1, seq2, param_list[0], param_list[1], param_list[2])
    with open(
            f"../alignment/ResultRevAI/TestScoringResult/{file_code}_match{param_list[0]}_mismatch{param_list[1]}_gap{param_list[2]}.csv",
            'w') as file:
        output = csv.writer(file)
        output.writerows([align12, align21, map12, map21])
    print(f"{file_code} has been written.\n")


if __name__ == "__main__":
    parameter_list = []
    for i in range(-10, 11):
        for j in range(-10, 11):
            for k in range(-10, 11):
                parameter_list.append([i, j, k])
    with Pool(12) as pool:
        pool.map(test_scoring, parameter_list)
