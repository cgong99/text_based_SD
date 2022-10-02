from genalog.text import alignment
from genalog.text import anchor
from TranscriptProcess import *
from alignment import Token

gt_txt = "But uh, yeah, it's still, it's still fairly it's still a fairly expensive call but &uh but it's a_lot less expensive if we do it that way than if we"
noise_txt = "but uh yeah it's still it's still fairly fairly expensive call but uh but it's a lot less expensive if we do it "

# # These two methods are interchangeable, but work best at different character length as mentioned above
# aligned_gt, aligned_noise = anchor.align_w_anchor(gt_txt, noise_txt, gap_char="@")
# print(f"Aligned ground truth: {aligned_gt}")
# print(f"Aligned noise:        {aligned_noise}")


aligned_gt, aligned_noise = alignment.align(gt_txt, noise_txt, gap_char="@")
# print(f"Aligned ground truth: {aligned_gt}")
# print(f"Aligned noise:        {aligned_noise}")


# Process the aligned strings to find out how the tokens are related
gt_to_noise_mapping, noise_to_gt_mapping = alignment.parse_alignment(aligned_gt, aligned_noise)
# print(f"gt_to_noise: {gt_to_noise_mapping}")
# print(f"noise_to_gt: {noise_to_gt_mapping}")


amazon_to_gt = {"spk_0": "A", "spk_1": "B"}
rev_to_gt = {0: "A", 1: "B"}


def whole_string(tokens: Token):
    res = ""
    for token in tokens:
        res = res + " " + token.value
    return res


def F1(gt_to_noise, noise_to_gt, gt_tokens, hyp_tokens):
    p = percision(noise_to_gt, gt_tokens, hyp_tokens)
    r = recall(gt_to_noise, gt_tokens, hyp_tokens)
    f1 = p * r / (p + r)
    print("P=", p)
    print("R=", r)
    print("F1=", f1)
    return f1


def percision(noise_to_gt, gt_tokens, hyp_tokens):
    correct_count = 0
    for i in range(len(noise_to_gt)):
        gt_index = noise_to_gt[i][0]
        if len(noise_to_gt[i]) < 1:
            print("unmatched")
        hyp_spk = hyp_tokens[i].spk_id
        gt_spk = gt_tokens[gt_index].spk_id
        # if gt_spk == amazon_to_gt[hyp_spk]:
        if gt_spk == rev_to_gt[hyp_spk]:
            correct_count += 1

    print(correct_count)
    return correct_count / len(noise_to_gt)


def recall(gt_to_noise, gt_tokens, hyp_tokens):
    correct_count = 0
    for i in range(len(gt_to_noise)):
        hyp_index = gt_to_noise[i][0]

        gt_spk = gt_tokens[i].spk_id
        hyp_spk = hyp_tokens[hyp_index].spk_id
        # if gt_spk == amazon_to_gt[hyp_spk]:
        if gt_spk == rev_to_gt[hyp_spk]:
            correct_count += 1
    return correct_count / len(gt_to_noise)


def check_cross(mapping):
    max_index = mapping[0][0]
    for match in mapping:
        if match[0] < max_index:
            print("CROSSING: ", match, max_index)


amazon = Amazon("CallHome_eval/amazon/4074.json")
tokens = amazon.get_token_list()

rev = RevAI("CallHome_eval/rev/4074_cut.json")
tokens = rev.get_token_list()

gt = CallHome("CallHome_eval/transcripts/4074.cha")
gt_tokens = gt.get_token_list()

gt_str = whole_string(gt_tokens)
hyp_str = whole_string(tokens)

# print(gt_str)


aligned_gt, aligned_noise = alignment.align(gt_str, hyp_str, gap_char="@")
# print(f"Aligned ground truth: {aligned_gt}")
# print(f"Aligned noise:        {aligned_noise}")

# Process the aligned strings to find out how the tokens are related
gt_to_noise_mapping, noise_to_gt_mapping = alignment.parse_alignment(aligned_gt, aligned_noise)
# print(f"gt_to_noise: {gt_to_noise_mapping}")
# print(f"noise_to_gt: {noise_to_gt_mapping}")
print("map length: ", len(noise_to_gt_mapping))
print("hyp tokens: ", len(tokens))

check_cross(gt_to_noise_mapping)
check_cross(noise_to_gt_mapping)

print(F1(gt_to_noise_mapping, noise_to_gt_mapping, gt_tokens, tokens))
