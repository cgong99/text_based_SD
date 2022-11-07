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

def writeAlignmentWithErrors(filepath, outputname):
  """read transcripts and generate alignment with errors in csv file

  Args:
      filepath (string): input CallHome transcript path
      outputname (string): output filename
  """
  spk1, spk2, output = getAlignedString(filepath)
  spk1, spk2, output = generateExtraWordsErrors(spk1, spk2, output)
  spk1, spk2, output = generateMissedWordsErrors(spk1, spk2, output)
  writeFile(outputname, spk1, spk2, output)


def getAlignedString(filepath):
  """generate truth alignment csv base on ground truth transcript
  """
  annotations = CallHome(filepath).get_file_annotation(with_utterances=True)
  spk_1 = []
  spk_2 = []
  output = []
  for i in range(int(len(annotations))):
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
  return spk_1, spk_2, output

def findNonGapEntry(arr):
  res = []
  for i in range(len(arr)):
    if arr[i] != "-":
      res.append(i)
  return res

def generateMissedWordsErrors(spk1, spk2, output, percent=8):
  output_nonGap = findNonGapEntry(output)
  error_num = int(len(output)*percent/100)
  print("missed words: ", error_num)
  index_list = np.random.randint(0,len(output_nonGap), error_num)
  for i in index_list:
    if spk1[i] != "-" or spk2[i] != "-":
      output[output_nonGap[i]] = "-"
  return spk1, spk2, output

def generateExtraWordsErrors(spk1, spk2, output, percent=2):
  error_num = int(len(output)*percent/100)
  spk1_nonGap = findNonGapEntry(spk1)
  index_list1 = np.random.randint(0,len(spk1_nonGap), error_num)
  for i in index_list1:
    spk1[spk1_nonGap[i]] = "-"
  index_list2 = np.random.randint(0,len(output), error_num)
  for i in index_list2:
    spk2[i] = "-"
  return spk1, spk2, output

def generateWrongSpelling(spk1, spk2, output, percent=6):
  error_num = int(len(output)*percent/100)
  

def writeFile(name, spk1, spk2, output):
  with open(name, "w") as f:
    writer = csv.writer(f)
    writer.writerows([spk1,spk2,output])

def generateErrors(name):
  """add gaps to the original alignment

  Args:
      name : genrated alignment
  """
  spk1, spk2, output = readAlignment(name)
  spk1, spk2, output = generateExtraWordsErrors(spk1, spk2, output)
  spk1, spk2, output = generateMissedWordsErrors(spk1, spk2, output)
  writeFile("4074_test.csv", spk1, spk2, output)


def alignAcc(gt_file, output_file):
  spk1_gt, spk2_gt = extractAlignmentArray(gt_file)
  spk1, spk2 = extractAlignmentArray(output_file)
  spk1_gt = spk1_gt[1:]
  spk2_gt = spk2_gt[1:]
  spk1 = spk1[1:]
  spk2 = spk2[1:]
  count = 0
  for i in range(len(spk1_gt)):
    if spk1[i] != spk1_gt[i]:
      count += 1
  for i in range(len(spk2_gt)):
    if spk2[i] != spk2_gt[i]:
      count += 1
  print("difference count: ", count)
  print("accuracy: ", ((len(spk1)+len(spk2))-count)/(len(spk1)+len(spk2)))

if __name__ == "__main__":
  writeAlignmentWithErrors("../data/CallHome_eval/transcripts/4074.cha", "4074_test1.csv")
  
  # alignAcc("4074_test.csv", "4074_disable_spk12_align_result.csv")
  