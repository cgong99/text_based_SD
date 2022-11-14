from Text_Based_SD.alignment.eval import Eval_3d
import csv

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


def computHypDER(utterance_list, hyp_align, spk_map_hyp2gt):
  """compute DER use each hyp utterance as segments

  Args:
      utterance_list (list): list of hyp spk_id and utterance [(spkid, "utterance")]
      hyp_align (list): [spk0-1, spk0-2, spk1-1, spk1-2, -, ...]
      spk_map (dict): {spk_0:A, spk_1:B}
  """
  hyp_index = 0
  numerator = 0
  denominator = 0
  for spk_utterance in utterance_list:
    spk = spk_utterance[0]
    utterance = spk_utterance[1]
    # get utterance spk_id
    spk_dict, hyp_index = getSpkAlignInfo(utterance, hyp_align, hyp_index) # get a dict stores start and end of aligned gt spk utterance
    Nref = len(spk_dict.keys())
    # Ncorrect compare spk_id with spk_dict.keys()
    # print(spk_map[spk])
    # print(spk_dict.keys())
    if spk_map_hyp2gt[spk] in spk_dict.keys():
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
  return der

def computeGTDER(utterance_list,spk1_align, spk2_align, hyp_align, spk_map_gt2hyp):
  """compute DER score using ground truth transcript utterances as segments

  Args:
      utterance_list (list):[(spk,start,end,utterance)...]
      spk1_align (list): _description_
      spk2_align (list): _description_
      hyp_align (list): _description_
      spk_map (dict): _description_
  """
  spk1_count, spk2_count = 0, 0
  numerator = 0
  denominator = 0
  for spk_utterance in utterance_list:
    gt_spk, utterance = spk_utterance[0], spk_utterance[3]
    gt_tokens = list(filter(lambda a: a!="", utterance.split(" "))) # split utterance into tokens and remove empty token
    utt_len = len(gt_tokens)
    if gt_spk == 'A':
      align = spk1_align[spk1_count:spk1_count+utt_len-1]
      spk1_count += utt_len
    elif gt_spk == 'B':
      align = spk2_align[spk2_count:spk2_count+utt_len-1]
      spk2_count += utt_len
    else:
      print("Error: unseen speaker\n")
      return
    align = list(filter(lambda a: a!= -1, align)) # remove gaps -1
    if not align: # if the entire utterance is gapped
      Ncorrect = 0
      Nhyp = 0
    else:
      start,end = int(min(align)), int(max(align)) # get the start and end index in hyp
      spk_set = getHypSegmentSpkSet(hyp_align=hyp_align, start=start, end=end)
      Nhyp = len(spk_set)
      if gt_spk in spk_set:
        Ncorrect = 1
      else:
        Ncorrect = 0
    Nref = 1
    numerator += utt_len*(max(1, Nhyp) - Ncorrect)
    denominator += utt_len*Nref
    # if (max(1, Nhyp) - Ncorrect) == 1:
    #   print("GT_SPK", gt_spk)
    #   print("Nhyp", Nhyp)
    #   print("Ncorrrect", Ncorrect)
    #   print("Set:", spk_set)
      
  der = numerator/denominator
  print(der)
  return der


def getHypSegmentSpkSet(hyp_align, start, end):
  """return a set of speaker id in the input hyp segment
  
  Args:
      hyp_align (list): alignment between hyp and gt
      start (int): start index
      end (int): end index

  Returns:
      set: a set contains hyp speaker id in this segment
  """
  spk_set = set()
  for i in range(start-1, end):
    if i >= len(hyp_align):
      print(start, " ", end)
      print(len(hyp_align))
    spk, index = getSpkIndex(hyp_align[i])
    if not spk:
      continue
    spk_set.add(spk)
  return spk_set

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


def checkRawLength():
  eval = Eval_3d(file_code=4093, type="Rev")
  utt = eval.hyp_raw_file.get_utterances_by_spkID()
  tokens = eval.hyp_tokens
  res = 0
  tk_count = 0
  print(len(tokens))
  for u in utt:
    # res += len(u[1].split(" ")[1:])
    res += len(list(filter(lambda a: a!="", u[1].split(" "))))
    u_tokens = list(filter(lambda a: a!="", u[1].split(" ")))
    for t in u_tokens:
      if str.lower(t) != tokens[tk_count].value:
        print("diff")
        print(tk_count)
        print("raw", t, "tk", tokens[tk_count].value)

      tk_count += 1
    # print(u[1].split(" "))
  print("raw:", res)


  u = utt[1]
  raw_tokens = list(filter(lambda a: a!="", u[1].split(" ")))
  print(len(raw_tokens))
  print(raw_tokens)
  for i in range(160,160+249):
    print(tokens[i].value, end=" ")
  print("\n")
  
def checkGTUtteranceLength():
  eval = Eval_3d(file_code=4093, type="Rev")
  utts = eval.gt_file.get_file_annotation(with_utterances=True)
  # print(utts[0][3].split(" "))
  # print(utts[1][3].split(" "))
  # for i in range(30):
  #   print(eval.gt_tokens[i].value, end=" ")
  token_count = 0
  for a in utts:
    utt = a[3]
    tokens = utt.split(' ')
    tokens=list(filter(lambda a: a!="", tokens))
    token_count += len(tokens)
  
  print(token_count)
  print(len(eval.gt_tokens))

def computeTwoDER(file_code:int,type:str):
  if type == "Amazon":
    spk_map_gt2hyp = {'A': "spk_0", 'B': "spk_1"}
    spk_map_hyp2gt = {"spk_0": "A", "spk_1": "B"}
  elif type == "Rev":
    spk_map_gt2hyp = {'A':0, 'B':1}
    spk_map_hyp2gt = {0: "A", 1:"B"}
  eval = Eval_3d(file_code=file_code, type=type)
  hyp_align = eval.getHyp2SpkAlignment()
  gt_utterances = eval.gt_file.get_file_annotation(with_utterances=True)
  hyp_utterances = eval.hyp_raw_file.get_utterances_by_spkID()
  spk1_align = eval.spk1_align
  spk2_align = eval.spk2_align
  gt_der = computeGTDER(utterance_list=gt_utterances,spk1_align=spk1_align,spk2_align=spk2_align,hyp_align=hyp_align,spk_map_gt2hyp=spk_map_gt2hyp)
  hyp_der = computHypDER(utterance_list=hyp_utterances,hyp_align=hyp_align,spk_map_hyp2gt=spk_map_hyp2gt)
  return gt_der, hyp_der

def writeDERcsv(type:str, output_file:str):
  # "ResultRevAI/Rev_3D_DER.csv"
  header = ["file",  "Precision_DER", "Recall_DER"]
  pool = ["4074", "4315", "4093", "4247", "4325", "4335", "4571", "4595"]
  with open(output_file, 'w') as file:
    output = csv.writer(file)
    output.writerow(header)
    for code in pool:
      print(code)
      gt_der, hyp_der = computeTwoDER(file_code=code, type=type)
      row = [code, hyp_der, gt_der]
      output.writerow(row)
      
    
  
if __name__ == "__main__":
  # Amazon_spk_map = {"spk_0": "A", "spk_1": "B"}
  # Rev_spk_map = {0: "A", 1:"B"}
  Rev_spk_map_gt2hyp = {'A':0, 'B':1}
  eval = Eval_3d(file_code=4074, type="Rev")
  # utterances = eval.hyp_raw_file.get_utterances_by_spkID()
  # # print(utterances)
  # hyp_align = eval.getHyp2SpkAlignment()

  # computeTwoDER(file_code=4093,type="Rev")
  # computeTwoDER(file_code=4074,type="Amazon")
  # writeDERcsv("Rev", "ResultRevAI/Rev_3D_DER.csv")
  writeDERcsv("Amazon", "ResultAmazon/Amazon_3D_DER.csv")