from Text_Based_SD.alignment.eval import Eval_3d


# get raw to ground truth alignment
# get each uttereance 
# each utterance alignment, get first aligned token to last aligned token in gt transcripts.
# [a-1,....,a-15, b-10, b-12, a-19], get a-[1-19] b[10-12]

def getHyp2SpkAlignment(spk1ToTarget, spk2ToTarget, hyp_len):
  """base on two speakers alignment, generate hyp_to_spk alignment

  Returns:
      [a-1, a-2, b-1, b-2,...]
  """
  hyp_align = [-1 for i in range(hyp_len)]
  for i, align in enumerate(spk1ToTarget):
    if align > 0:
      hyp_align[align] = "a-" + str(i)
      
  for i, align in enumerate(spk2ToTarget):
    if align > 0:
      hyp_align[align] = "b-" + str(i)
      
      
  return hyp_align
  

def computDER(utterance_list, hyp_align, spk_map):
  # utterances: [(spkid, "utterance")]
  # hyp_align: [a-1, a-2, b-1, b-2, -, ...]
  hyp_index = 0
  numerator = 0
  denominator = 0
  for spk_utterance in utterance_list:
    spk = spk_utterance[0]
    utterance = spk_utterance[1]
    # get utterance spk_id
    spk_dict, hyp_index = getSpkAlignInfo(utterance, hyp_align, hyp_index)
    Nref = len(spk_dict.keys())
    # Ncorrect compare spk_id with spk_dict.keys()
    # print(spk_map[spk])
    # print(spk_dict.keys())
    if spk_map[spk] in spk_dict.keys():
      Ncorrect = 1
    else:
      Ncorrect = 0
      
    # print(Nref)
    # print(Ncorrect)
    utt_len = getUtteranceLen(utterance)
    numerator += utt_len*(max(1, Nref) - Ncorrect)
    denominator += utt_len*Nref
  
  der = numerator/denominator
  print(der)

    

def getSpkAlignInfo(utterance, hyp_align, hyp_index):
  utt_len = getUtteranceLen(utterance)
  spk_dict = {}
  for i in range(utt_len):
    spk, index = getSpkIndex(hyp_align[hyp_index])
    if not spk: # no aligned speaker to gt
      hyp_index += 1
      continue
    elif spk not in spk_dict:
      spk_dict[spk] = (index, index)
    else:
      start = spk_dict[spk][0]
      end = spk_dict[spk][1]
      spk_dict[spk] = (min(index, start), max(index, end))
    hyp_index += 1
  return spk_dict, hyp_index
  
def getUtteranceLen(utterance):
  # TODO: may need to use the ref length?
  return len(utterance.split(" ")[1:])

def getSpkIndex(align: str):
  # input "a-1"
  if align == "-":
    return None, None
  res = align.split("-")
  return res[0], res[1]


def rawLength(utt):
  res = 0
  for u in utt:
    res += len(u[1].split(" ")[1:])
    # print(u[1].split(" "))
  print("raw:", res)

if __name__ == "__main__":
  Amazon_spk_map = {"spk_0": "A", "spk_1": "B"}
  Rev_spk_map = {0: "A", 1:"B"}
  eval = Eval_3d(file_code=4074, type="Rev")
  utterances = eval.hyp_raw_file.get_utterances_by_spkID()
  # print(utterances)
  hyp_align = eval.getHyp2SpkAlignment()
  # print(utterances[:3])
  computDER(utterance_list=utterances, hyp_align=hyp_align, spk_map=Rev_spk_map)


  # for i in range(50):
  #   print(eval.hyp_tokens[i].value, end=" ")
  # print("\n")
  # print(utterances[0][1].split(" "))
  # print(utterances[1][1].split(" "))
  # print(len(eval.hyp_tokens))
  # rawLength(utterances)