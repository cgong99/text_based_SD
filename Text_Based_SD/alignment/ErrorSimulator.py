import copy
import random


def get_simulation(token_list: list[str], speaker_list: list[str], deletion: int, addition: int, insertion: int,
                   substitution: int):
    """
    Produce simulated ASR transcript with 4 types of error. Multiparty is supported.

    deletion: Delete 1 token from hypothesis token and replace it with a gap, alignment remain unchanged.

    addition: Add 1 token to hypothesis token, a gap will be added into all the reference token at the same index,
    in order to maintain the correct alignment all the alignment index will be pushed back for 1 position by adding 1 to
    all index afterwards if index not -1.
    Currently, the added token is fixed to string "++addition++".
    In the future, the added token should be customizable or automatically deduced according to context.

    insertion: This error is to simulate the overlapping issue. Select a segment of tokens at the beginning of an
    utterance and insert it into the end of the utterance at front. Their speaker labels and alignment index are also
    switched in order to maintain a correct reference to the switched tokens.
    Currently, switch two tokens at the boundary of two utterances, also the position of their speaker labels and
    alignment index are switched in order to maintain the correct reference to the switched tokens.
    In the future, the length of tokens to be inserted into the front, the position of inserting should all be
    customizable or automatically deduced according to context.

    substitution: Change 1 token in hypothesis token to a variant that is close in terms of sound or pronunciation. The
    alignment index remains unchanged.
    Currently, for simplicity the token will be modified as ""**original-token**".
    In the future, the change of token should be customizable or automatically deduced according to context.

    Procedure:
    1. Generate hypothesis token and speaker label list by having a deepcopy of the reference token and speaker label
       list passed in to the function. At this stage they are identical to the reference.
    2. Produce a list of speaker label reference with no repetition, this creates a unique index for each speaker.
    3. Add insertion error to the hypothesis token and speaker.
    4. Separate the reference token and create correct alignment index.
    5. Randomly select tokens and positions for introducing different types of errors. These tokens and positions cannot
       be overlapped.
    6. Introduce substitution error.
    7. Introduce deletion error.
    8. Introduce addition error.

    :param token_list: List of reference tokens. This should be produced by direct concatenation of reference utterances
    :param speaker_list: List of speaker labels that has correct mapping to the token_list
    :param deletion: Number of tokens that are going to have deletion error
    :param addition: Number of tokens that are going to have addition error
    :param insertion: Number of insertion errors. Currently, this number cannot exceed the number of speaker changes.
    :param substitution: Number of tokens that are going to have addition error
    :return: A tuple consists of: a list of unique speaker labels, a 2d-list of reference tokens separated according to
    speakers, a list of hypothesis tokens, a 2d-list of index mapping from reference to hypothesis
    """
    if len(token_list) != len(speaker_list):
        raise Exception("length of token list is not matched with the speaker list.")
    if deletion + addition + substitution >= len(token_list):
        raise Exception("error is more than total number of token.")
    token_num = len(token_list)
    hypo_token = copy.deepcopy(token_list)
    hypo_speaker = copy.deepcopy(speaker_list)
    speaker_label = list(set(hypo_speaker))  # remove duplicate speaker label

    # introduce token insertion error
    # detect speaker change position, the index is the start position of each utterance except the very first one
    speaker_change = [i for i in range(1, len(hypo_speaker)) if hypo_speaker[i - 1] != hypo_speaker[i]]
    if insertion <= len(speaker_change):  # if number of overlapped tokens is less than number of speaker change
        insertion_index = random.sample(speaker_change, insertion)  # randomly choose speaker change position
        for index in insertion_index:
            hypo_token[index - 1], hypo_token[index] = hypo_token[index], hypo_token[index - 1]
            hypo_speaker[index - 1], hypo_speaker[index] = hypo_speaker[index], hypo_speaker[index - 1]
    else:
        raise Exception("insertion error token number is greater than speaker change number.")

    # generate true alignment and reference tokens split according to speakers
    align_index = []
    ref_token = []
    for i in range(len(speaker_label)):
        align_index.append([])
        ref_token.append([])
        for j in range(token_num):
            if hypo_speaker[j] == speaker_label[i]:
                align_index[i].append(j)
                ref_token[i].append(hypo_token[j])
            else:
                align_index[i].append(-1)
                ref_token[i].append('-')

    print(speaker_label)
    print(align_index)
    print(ref_token)
    print(hypo_token)

    error_index = random.sample(range(token_num), (substitution + deletion + addition))

    # introduce token substitution error
    # currently the substituted word is marked as **original-word** in hypothesis text, the alignment stays the same
    substitution_index = error_index[0:substitution]
    for index in substitution_index:
        hypo_token[index] = "**" + hypo_token[index] + "**"

    # introduce token deletion error
    deletion_index = error_index[substitution:substitution + deletion]
    for index in deletion_index:
        speaker = hypo_speaker[index]
        label_index = speaker_label.index(speaker)
        hypo_token[index] = '-'
        align_index[label_index][index] = -1

    # introduce token addition error
    # currently the addition word is marked as ++addition++ in hypothesis text
    addition_index = error_index[substitution + deletion:substitution + deletion + addition]
    for index in addition_index:
        hypo_token.insert(index, "++addition++")
        token_num += 1
        for i in range(len(align_index)):
            ref_token[i].insert(index, '-')
            align_index[i].insert(index, -1)
            for j in range(index, token_num):
                if align_index[i][j] != -1:
                    align_index[i][j] += 1

    print(speaker_label)
    print(align_index)
    print(ref_token)
    print(hypo_token)

    return speaker_label, ref_token, hypo_token, align_index


if __name__ == "__main__":
    token_list = ["I", "am", "a", "fish", "Good", "idea", "You", "are", "too"]
    speaker_list = ["A", "A", "A", "A", "B", "B", "C", "C", "C"]
    get_simulation(token_list, speaker_list, 2, 2, 1, 1)
