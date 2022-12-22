import copy

import numpy as np
from numba import jit
from numpy import ndarray
from itertools import combinations
from typing import Iterable

from NeedlemanWunsch import edit_distance


def compare(reference: str, hypothesis_list: list[str]) -> int:
    """
    fully match: 2, non-fully match: 1, others (gap or mismatch): -1
    Args:
        reference: single token for reference text
        *hypothesis_list: iterable of hypothesis token, each token represent one speaker
    Returns: score as integer
    """
    gap = '-'
    hypothesis = gap
    for token in hypothesis_list:
        if token == gap:
            continue
        else:
            if hypothesis == gap:
                hypothesis = token
            else:
                return -1
    if reference == gap:
        return -1
    else:
        if reference == hypothesis:
            return 2
        elif edit_distance(reference, hypothesis) < 2:
            return 1
        else:
            return -1


def get_sequence_position_list(speaker_sequence_length: int) -> list[tuple]:
    sequence_position = []
    for i in range(1, speaker_sequence_length + 1):
        for combination in combinations(range(speaker_sequence_length), i):
            sequence_position.append(combination)
    return sequence_position


def get_current_index(position: tuple | list, matrix_size: list[int]) -> tuple:
    sequence_num = len(matrix_size)
    current = [0 if i not in position else 1 for i in range(sequence_num)]
    while True:
        yield tuple(current)
        current[position[len(position) - 1]] += 1
        for i in range(len(position) - 1, -1, -1):
            if current[position[0]] == matrix_size[position[0]]:
                return
            if current[position[i]] == matrix_size[position[i]]:
                current[position[i - 1]] += 1
                current[position[i]] = 1


def get_parameter_index_list(sequence_position: Iterable, current_index: Iterable) -> list[tuple]:
    # add all gap (only one token) situations
    parameter_index_list = []
    for index in sequence_position:
        single_token_index = [j if i != index else j - 1 for i, j in enumerate(current_index)]
        parameter_index_list.append(tuple(single_token_index))
        if 0 in sequence_position and index != 0:
            double_token_index = copy.deepcopy(single_token_index)
            double_token_index[0] -= 1
            parameter_index_list.append(tuple(double_token_index))
    return parameter_index_list


def get_compare_parameter(current_index: tuple | list, parameter_index: tuple | list, speaker_sequence: list[list[str]]):
    gap = '-'
    compare_parameter = []
    for i in range(len(current_index)):
        if current_index[i] != parameter_index[i]:
            compare_parameter.append(speaker_sequence[i][parameter_index[i]])
        else:
            compare_parameter.append(gap)
    return compare_parameter[0], compare_parameter[1:]


def multi_sequence_alignment(reference: list[str], hypothesis: list[list[str]]):
    speaker_sequence = copy.deepcopy(hypothesis)
    speaker_sequence.insert(0, copy.deepcopy(reference))
    matrix_size = [len(speaker) + 1 for speaker in speaker_sequence]
    score = np.zeros(tuple(matrix_size), dtype="int32")

    for sequence_position in get_sequence_position_list(len(speaker_sequence)):
        for current_index in get_current_index(sequence_position, matrix_size):
            parameter = []
            for parameter_index in get_parameter_index_list(sequence_position, current_index):
                ref, hypo = get_compare_parameter(current_index, parameter_index, speaker_sequence)
                parameter.append(score[parameter_index] + compare(ref, hypo))
            score[current_index] = max(parameter)

    # backtracking
    align_sequence = [[] for _ in speaker_sequence]
    mappings = [np.zeros(i) for i in matrix_size]
    current_index = copy.deepcopy(matrix_size)
    while sum(current_index) > 0:
        sequence_position = tuple([i for i, j in enumerate(current_index) if j != 0])
        for parameter_index in get_parameter_index_list(sequence_position, current_index):
            ref, hypo = get_compare_parameter(current_index, parameter_index, speaker_sequence)
            if score[tuple(current_index)] == compare(ref, hypo) + score[parameter_index]:
                # append word to aligned sequence
                align_sequence[0].append(ref)
                for i in range(1, len(align_sequence)):
                    align_sequence[i].append(hypo[i - 1])
    return align_sequence


if __name__ == "__main__":
    ref = ["I", "am", "a", "fish", "Are", "you"]
    hypo = [["I", "am", "a", "fish"], ["Are", "you"]]
    print(multi_sequence_alignment(ref, hypo))
