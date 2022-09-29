from genalog.text import alignment
from genalog.text import anchor

gt_txt = "But uh, yeah, it's still, it's still fairly it's still a fairly expensive call but &uh but it's a_lot less expensive if we do it that way than if we"
noise_txt = "but uh yeah it's still it's still fairly fairly expensive call but uh but it's a lot less expensive if we do it "

# # These two methods are interchangeable, but work best at different character length as mentioned above
# aligned_gt, aligned_noise = anchor.align_w_anchor(gt_txt, noise_txt, gap_char="@")
# print(f"Aligned ground truth: {aligned_gt}")
# print(f"Aligned noise:        {aligned_noise}")


aligned_gt, aligned_noise = alignment.align(gt_txt, noise_txt, gap_char="@")
print(f"Aligned ground truth: {aligned_gt}")
print(f"Aligned noise:        {aligned_noise}")


# Process the aligned strings to find out how the tokens are related
gt_to_noise_mapping, noise_to_gt_mapping = alignment.parse_alignment(aligned_gt, aligned_noise)
print(f"gt_to_noise: {gt_to_noise_mapping}")
print(f"noise_to_gt: {noise_to_gt_mapping}")


