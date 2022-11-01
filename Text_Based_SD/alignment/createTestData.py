from os import read
import pandas as pd
import numpy as np
import csv
from Text_Based_SD.data.TranscriptProcess import CallHome
from NeedlemanWunsch3D import *


def readAlignment(name):
  f = open(name)
  lines = f.readlines()
  spk1 = lines[0].replace("\n", "").split(",")
  spk2 = lines[1].replace("\n", "").split(",")
  output = lines[2].replace("\n", "").split(",")
  # print(spk2)
  return spk1, spk2, output

def getTokenSeq(filename):
  f = open(filename)
  lines = f.readlines()
  # spk1_tokens, spk2_tokens, output_tokens = [],[],[]
  spk1 = lines[0].replace("\n", "").split(",")
  spk2 = lines[1].replace("\n", "").split(",")
  output = lines[2].replace("\n", "").split(",")
  spk1_tokens = [token for token in spk1 if token != "-"]
  spk2_tokens = [token for token in spk2 if token != "-"]
  output_tokens = [token for token in output if token != "-"]
  return spk1_tokens, spk2_tokens, output_tokens

def extractAlignmentArray(name):
  spk1, spk2, output = readAlignment(name)
  print(len(spk1), len(spk2), len(output))
  spk1_to_output, spk2_to_output = [], []
  outputGap = 0
  for i, token in enumerate(output):
    if token == "-":
      outputGap += 1
    if spk1[i] != "-" and token != "-":
      spk1_to_output.append(i-outputGap)
    elif spk1[i] != "-" and token == "-":
      spk1_to_output.append(-1)
      
    if spk2[i] != "-" and token != "-":
      spk2_to_output.append(i-outputGap)
    elif spk2[i] != "-" and token == "-":
      spk2_to_output.append(-1)
  
  return spk1_to_output, spk2_to_output

def writeNormalAlignment():
  annotations = CallHome("../data/CallHome_eval/transcripts/4074.cha").get_file_annotation(with_utterances=True)
  spk_1 = []
  spk_2 = []
  output = []
  for i in range(int(len(annotations)/2)):
    annotation = annotations[i]
    spk = annotation[0]
    utterance = annotation[3].split()
    if spk == "A":
      for token in utterance:
        spk_1.append(token)
        spk_2.append("-")
        output.append(token)
    elif spk == "B":
      for token in utterance:
        spk_1.append("-")
        spk_2.append(token)
        output.append(token)
  with open("test1.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows([spk_1,spk_2,output])

def generateMissedWordsErrors(spk1, spk2, output, num=50):
  index_list = np.random.randint(0,len(output), num)
  for i in index_list:
    output[i] = "-"
  return spk1, spk2, output

def generateExtraWordsErrors(spk1, spk2, output, num=10):
  index_list1 = np.random.randint(0,len(output), num)
  for i in index_list1:
    spk1[i] = "-"
  index_list2 = np.random.randint(0,len(output), num)
  for i in index_list2:
    spk2[i] = "-"
  
  return spk1, spk2, output

def writeFile(name, spk1, spk2, output):
  with open(name, "w") as f:
    writer = csv.writer(f)
    writer.writerows([spk1,spk2,output])

def generateErrors(name):
  spk1, spk2, output = readAlignment(name)
  spk1, spk2, output = generateMissedWordsErrors(spk1, spk2, output)
  writeFile("4074_test.csv", spk1, spk2, output)

if __name__ == "__main__":
  # generateErrors("test1.csv")
  # spk1, spk2, output = readAlignment("test1.csv")
  # spk1, spk2, output = generateMissedWordsErrors(spk1, spk2, output)
  # writeFile("4074_test.csv", spk1, spk2, output)
  
  spk1_gt, spk2_gt = extractAlignmentArray("4074_test.csv")
  spk1, spk2 = extractAlignmentArray("4074_disable_spk12_align_result.csv")
  # spk1 = spk1[1:]
  # spk2 = spk2[1:]
  print(len(spk1_gt), len(spk1))
  print(len(spk2_gt), len(spk2))
  print(spk1[150:200])
  print(spk1_gt[150:200])
  count = 0
  for i in range(len(spk1_gt)):
    if spk1[i] != spk1_gt[i]:
      count += 1
  for i in range(len(spk2_gt)):
    if spk2[i] != spk2_gt[i]:
      count += 1
  print("difference count: ", count)
  print("accuracy: ", ((len(spk1)+len(spk2))-count)/(len(spk1)+len(spk2)))
  
  # seq1, seq2, target = getTokenSeq("4074_test.csv")
  # align1, align2, align3, align2_to_align1, align3_to_align1 = backtrack(target, seq1, seq2, get_scoring_matrix_3d(target, seq1, seq2, 4074), 4074)
                                                                                           
  # with open(f"4074_disable_spk12_align_result.csv", 'w') as file:
  #   output = csv.writer(file)
  #   output.writerows([align2, align3, align1, align2_to_align1, align3_to_align1])


  