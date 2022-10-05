import numpy as np
from numpy import ndarray


def edit_distance(token1: str, token2: str) -> int:
    """
    Compute the Levenshtein distance between two string
    :param token1: token as string
    :param token2: token as string
    :return: Levenshtein distance in int
    """
    matrix = np.zeros(len(token1) + 1, len(token2) + 1)
    for i in range(1, len(token1) + 1):
        matrix[i][0] = i
    for j in range(1, len(token2) + 1):
        matrix[0][j] = j
    for i in range(1, len(token1)):
        for j in range(1, len(token2)):
            substitution = 0 if token1[i - 1] == token2[j - 1] else 1
            matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + substitution)
    return matrix[len(token1)][len(token2)]


def compare(token1: str, token2: str, distance_bound: int = 2) -> int:
    """
    Compare two string and determine if they are match,
    if the levenshtein distance is below the distance bound then count as match
    :param token1: token as string
    :param token2: token as string
    :param distance_bound: bound of levenshtein distance
    :return: integer represent the score of match, mis-match, or a gap
    """
    match = 2
    mis_match = -1
    gap = -1
    if edit_distance(token1, token2) <= distance_bound:
        return match
    elif token1 == '-' or token2 == '-':
        return gap
    else:
        return mis_match


def get_scoring_matrix(seq1: list[str], seq2: list[str]) -> ndarray:
    """
    Compute the scoring matrix for Needleman-Wunsch algorithm
    :param seq1: list of words from the dialogue
    :param seq2: list of words from the dialogue
    :return: 2d numpy array as the scoring matrix
    """
    gap = -1
    # compute scoring matrix
    score = np.zeros((len(seq1) + 1, len(seq2) + 1))
    for i in range(0, len(seq1) + 1):
        score[i][0] = gap * i
    for j in range(0, len(seq2) + 1):
        score[0][j] = gap * j
    for i in range(1, len(seq1) + 1):
        for j in range(1, len(seq2) + 1):
            score[i][j] = max(score[i - 1][j - 1] + compare(seq1[i - 1], seq2[j - 1]),
                              score[i - 1][j] + gap, score[i][j - 1] + gap)
    return score


def backtrack(seq1: str, seq2: str, score: ndarray) -> tuple[list[str], ndarray, list[str], ndarray]:
    """
    Backtrack according to the scoring matrix to get the alignment result
    :param seq1: list of words from the dialogue, same as get scoring matrix
    :param seq2: list of words from the dialogue, same as get scoring matrix
    :param score: the scoring matrix from get scoring matrix
    :return: a tuple containing the list of aligned two sequence and their mapping to each other result
    """
    gap = -1
    i = len(seq1)
    j = len(seq2)
    align1 = []
    align1_to_align2 = np.zeros(i + 1)
    align2 = []
    align2_to_align1 = np.zeros(j + 1)
    while i > 0 and j > 0:
        if score[i][j] == score[i - 1][j - 1] + compare(seq1[i - 1], seq2[j - 1]):
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
