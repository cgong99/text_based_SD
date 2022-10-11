import timeit
from tokenize import String
import numpy as np
from numpy import ndarray
from typing import List, Tuple
from numba import jit

# from sklearn.metrics import get_scorer
from NeedlemanWunsch import edit_distance, get_scoring_matrix, compare
from Text_Based_SD.data.TranscriptProcess import RevAI, CallHome
import csv


@jit(nopython=True)
def score(token1, token2, token3) -> int:
  """return compare score for three tokens

    Args:
        token1 (string):
        token2 (string):
        token3 (string):

  Args:
      token1 (string): from target sequence
      token2 (string): from speaker 1
      token3 (string): from speaker 2

  Returns:
      int: score
  """
  if token1 != '-' and token2 != '-' and token3 == '-' and  edit_distance(token1, token2) < 2:
    return 2
  elif token1 != '-' and token3 != '-' and token2 == '-' and edit_distance(token1,token3) < 2:
    return 2
  if token1 != '-' and token2 != '-' and token3 != '-':
    return 0
  return -2

class MultiSeqAlign:
  def __init__(self, target, seq1, seq2) -> None:
    """initialize a 3-d matrix's 3 surfaces, and stores the three sequence
        target: m*1 x dimiension
        seq1: n*1 y dimiension
        seq2: d*1 z dimiension
    Args:
        seq1 (List):
        seq2 (List):
        seq3 (List):
    """
    self.gap = -1
    self.target, self.seq1, self.seq2 = target, seq1, seq2
    self.m, self.n, self.d = len(target)+1, len(seq1)+1, len(seq2)+1
    self.matrix = np.zeros((self.m, self.n, self.d))
    self.setupMatrix()

  def setupMatrix(self):
    """init 3 edges with gap values, and compute 3 surface using pairwise alignment
    """
    for i in range(self.m):
      self.matrix[i,0,0] = self.gap * i
    for i in range(self.n):
      self.matrix[0,i,0] = self.gap * i
    for i in range(self.d):
      self.matrix[0,0,i] = self.gap * i
    # setup xy
    self.setupSurface("xy")
    # setup yz
    self.setupSurface("yz")
    # setup xz
    self.setupSurface("xz")

  def setupSurface(self, axes: String):
    """setup one surface of a matrix based on the passed in axes, computing a 2d sequence alignment table
    Args:
        axes: "xy", "yz" or "xz"
    """
    x,y,z = 1,1,1
    table = None
    if axes == 'xy':
      x = self.m
      y = self.n
      table = get_scoring_matrix(seq1=self.target, seq2=self.seq1, gap=self.gap)
    elif axes == 'yz':
      y = self.n
      z = self.d
      table = get_scoring_matrix(seq1=self.seq1, seq2=self.seq2, gap=self.gap)
    elif axes == 'xz':
      x = self.m
      z = self.d
      table = get_scoring_matrix(seq1=self.target, seq2=self.seq2, gap=self.gap)
    else:
      print("dimension error: ", axes, " should be xy yz or xz")

    # the unselected dimension only contain 0
    for i in range(x):
      for j in range(y):
        for k in range(z):
          if axes == 'xy':
            self.matrix[i,j,k] = table[i,j]
          elif axes == 'yz':
            self.matrix[i,j,k] = table[j,k]
          elif axes == 'xz':
            self.matrix[i,j,k] = table[i,k]

  def compute_matrix(self):
    for i in range(1, self.m):
      for j in range(1, self.n):
        for k in range(1, self.d):
          #self.matrix[i,j,k] =
          xi = self.target[i-1]
          yj = self.seq1[j-1]
          zk = self.seq2[k-1]
          xGap = score('-',yj,zk) + self.matrix[i,j-1,k-1]
          yGap = score(xi,'-',zk) + self.matrix[i-1,j,k-1]
          zGap = score(xi,yj,'-') + self.matrix[i-1,j-1,k]
          yzGap = score(xi,'-','-') + self.matrix[i-1,j,k]
          xzGap = score('-',yj,'-') + self.matrix[i,j-1,k]
          xyGap = score('-','-',zk) + self.matrix[i,j,k-1]
          allMatch = score(xi,yj,zk) + self.matrix[i-1,j-1,k-1]
          self.matrix[i,j,k] = max([xGap, yGap, zGap, yzGap, xyGap, xzGap, allMatch])
    print("compute done")

  def backtrack(self):
    i = self.m - 1
    j = self.n - 1
    k = self.d - 1
    target, seq1, seq2 = [], [], []
    while i > 0 and j > 0 and k > 0:
      xi = self.target[i-1]
      yj = self.seq1[j-1]
      zk = self.seq2[k-1]
      if self.matrix[i,j,k] == score('-',yj,zk) + self.matrix[i,j-1,k-1]:
        target.append('-')
        seq1.append(yj)
        seq2.append(zk)
        j -= 1
        k -= 1
      elif self.matrix[i,j,k] == score(xi,'-',zk) + self.matrix[i-1,j,k-1]:
        target.append(xi)
        seq1.append('-')
        seq2.append(zk)
        i -= 1
        k -= 1
      elif self.matrix[i,j,k] == score(xi,yj,'-') + self.matrix[i-1,j-1,k]:
        target.append(xi)
        seq1.append(yj)
        seq2.append('-')
        i -= 1
        j -= 1
      elif self.matrix[i,j,k] == score(xi,'-','-') + self.matrix[i-1,j,k]:
        target.append(xi)
        seq1.append('-')
        seq2.append('-')
        i -= 1
      elif self.matrix[i,j,k] == score('-',yj,'-') + self.matrix[i,j-1,k]:
        target.append('-')
        seq1.append(yj)
        seq2.append('-')
        j -= 1
      elif self.matrix[i,j,k] == score('-','-',zk) + self.matrix[i,j,k-1]:
        target.append('-')
        seq1.append('-')
        seq2.append(zk)
        k -= 1
      elif self.matrix[i,j,k] == score(xi,yj,zk) + self.matrix[i-1,j-1,k-1]:
        target.append(xi)
        seq1.append(yj)
        seq2.append(zk)
        i -= 1
        j -= 1
        k -= 1

    # one of the dimension is 0, on a surface now
    while i > 0 and j > 0:
      xi = self.target[i-1]
      yj = self.seq1[j-1]
      if self.matrix[i,j,k] == compare(xi, yj) + self.matrix[i-1,j-1,k]:  # 2d using NeedlemanWunsch's compare
        target.append(xi)
        seq1.append(yj)
        seq2.append('-')
        i -= 1
        j -= 1
      elif self.matrix[i,j,k] == self.matrix[i-1,j,k] + self.gap:     # self.gap computes 2d pairwise alignment
        target.append(xi)
        seq1.append('-')
        seq2.append('-')
        i -= 1
      elif self.matrix[i,j,k] == self.matrix[i,j-1,k] + self.gap:
        target.append('-')
        seq1.append(yj)
        seq2.append('-')
        j -= 1

    while i > 0 and k > 0:
      xi = self.target[i-1]
      zk = self.seq2[k-1]
      if self.matrix[i,j,k] == compare(xi, zk) + self.matrix[i-1,j,k-1]: # 2d using NeedlemanWunsch's compare
        target.append(xi)
        seq1.append('-')
        seq2.append(zk)
        i -= 1
        k -= 1
      elif self.matrix[i,j,k] == self.matrix[i-1,j,k] + self.gap:
        target.append(xi)
        seq1.append('-')
        seq2.append('-')
        i -= 1
      elif self.matrix[i,j,k] == self.matrix[i,j,k-1] + self.gap:
        target.append('-')
        seq1.append('-')
        seq2.append(zk)
        k -= 1

    while j > 0 and k > 0:
      yj = self.seq1[j-1]
      zk = self.seq2[k-1]
      if self.matrix[i,j,k] == compare(yj,zk) + self.matrix[i,j-1,k-1]: # 2d using NeedlemanWunsch's compare
        target.append('-')
        seq1.append(yj)
        seq2.append(zk)
        j -= 1
        k -= 1
      elif self.matrix[i,j,k] == self.matrix[i,j-1,k] + self.gap:
        target.append('-')
        seq1.append(yj)
        seq2.append('-')
        j -= 1
      elif self.matrix[i,j,k] == self.matrix[i,j,k-1] + self.gap:
        target.append('-')
        seq1.append('-')
        seq2.append(zk)
        k -= 1

    while i > 0:
      xi = self.target[i-1]
      target.append(xi)
      seq1.append('-')
      seq2.append('-')
      i -= 1

    while j > 0:
      yj = self.seq1[j-1]
      target.append('-')
      seq1.append(yj)
      seq2.append('-')
      j -= 1

    while k > 0:
      zk = self.seq2[k-1]
      target.append('-')
      seq1.append('-')
      seq2.append(zk)
      k -= 1
    return target[::-1], seq1[::-1], seq2[::-1]

@jit(nopython=True)
def compare_3d(token1, token2, token3) -> int:
  if token1 != '-' and token2 != '-' and token3 == '-' and edit_distance(token1, token2) < 2:
    return 2
  elif token1 != '-' and token3 != '-' and token2 == '-' and edit_distance(token1, token3) < 2:
    return 2
  if token1 != '-' and token2 != '-' and token3 != '-':
    return 0
  return -2

@jit(nopython=True)
def get_scoring_matrix_3d(seq1: list[str], seq2: list[str], seq3: list[str]) -> ndarray:
  gap = -1
  count = 0
  score = np.zeros((len(seq1) + 1, len(seq2) + 1, len(seq3) + 1))
  for i in range(0, len(seq1) + 1):
    score[i][0][0] = gap * i
  for j in range(0, len(seq2) + 1):
    score[0][j][0] = gap * j
  for k in range(0, len(seq3) + 1):
    score[0][0][k] = gap * k
  for i in range(1, len(seq1) + 1):
    for j in range(1, len(seq2) + 1):
      for k in range(1, len(seq3) + 1):
        if count % 5000000 == 0:
          print(f"check point, i={i}, j={j}, k={k}")
        spec1 = score[i][j - 1][k - 1] + compare_3d('-', seq2[j - 1], seq3[k - 1])
        spec2 = score[i - 1][j][k - 1] + compare_3d(seq1[i - 1], '-', seq3[k - 1])
        spec3 = score[i - 1][j - 1][k] + compare_3d(seq1[i - 1], seq2[j - 1], '-')
        spec4 = score[i - 1][j][k] + compare_3d(seq1[i - 1], '-', '-')
        spec5 = score[i - 1][j][k - 1] + compare_3d('-', seq2[j - 1], '-')
        spec6 = score[i][j][k - 1] + compare_3d('-', '-', seq3[k - 1])
        spec7 = score[i - 1][j - 1][k - 1] + compare_3d(seq1[i - 1], seq2[j - 1], seq3[k - 1])
        score[i][j][k] = max(spec1, spec2, spec3, spec4, spec5, spec6, spec7)
        count += 1
  return score


@jit(nopython=True)
def backtrack(seq1, seq2, seq3, matrix):
  gap = -1
  i = len(seq1)
  j = len(seq2)
  k = len(seq3)
  count = 0
  align1, align2, align3 = [], [], []
  while i > 0 and j > 0 and k > 0:
    if count % 5000000 == 0:
      print(f"check point, i={i}, j={j}, k={k}")
    xi = seq1[i - 1]
    yj = seq2[j - 1]
    zk = seq3[k - 1]
    if matrix[i, j, k] == score('-', yj, zk) + matrix[i, j - 1, k - 1]:
      align1.append('-')
      align2.append(yj)
      align3.append(zk)
      j -= 1
      k -= 1
    elif matrix[i, j, k] == score(xi, '-', zk) + matrix[i - 1, j, k - 1]:
      align1.append(xi)
      align2.append('-')
      align3.append(zk)
      i -= 1
      k -= 1
    elif matrix[i, j, k] == score(xi, yj, '-') + matrix[i - 1, j - 1, k]:
      align1.append(xi)
      align2.append(yj)
      align3.append('-')
      i -= 1
      j -= 1
    elif matrix[i, j, k] == score(xi, '-', '-') + matrix[i - 1, j, k]:
      align1.append(xi)
      align2.append('-')
      align3.append('-')
      i -= 1
    elif matrix[i, j, k] == score('-', yj, '-') + matrix[i, j - 1, k]:
      align1.append('-')
      align2.append(yj)
      align3.append('-')
      j -= 1
    elif matrix[i, j, k] == score('-', '-', zk) + matrix[i, j, k - 1]:
      align1.append('-')
      align2.append('-')
      align3.append(zk)
      k -= 1
    elif matrix[i, j, k] == score(xi, yj, zk) + matrix[i - 1, j - 1, k - 1]:
      align1.append(xi)
      align2.append(yj)
      align3.append(zk)
      i -= 1
      j -= 1
      k -= 1

  # one of the dimension is 0, on a surface now
  while i > 0 and j > 0:
    xi = seq1[i - 1]
    yj = seq2[j - 1]
    if matrix[i, j, k] == compare(xi, yj) + matrix[i - 1, j - 1, k]:  # 2d using NeedlemanWunsch's compare
      align1.append(xi)
      align2.append(yj)
      align3.append('-')
      i -= 1
      j -= 1
    elif matrix[i, j, k] == matrix[i - 1, j, k] + gap:  # self.gap computes 2d pairwise alignment
      align1.append(xi)
      align2.append('-')
      align3.append('-')
      i -= 1
    elif matrix[i, j, k] == matrix[i, j - 1, k] + gap:
      align1.append('-')
      align2.append(yj)
      align3.append('-')
      j -= 1

  while i > 0 and k > 0:
    xi = seq1[i - 1]
    zk = seq3[k - 1]
    if matrix[i, j, k] == compare(xi, zk) + matrix[i - 1, j, k - 1]:  # 2d using NeedlemanWunsch's compare
      align1.append(xi)
      align2.append('-')
      align3.append(zk)
      i -= 1
      k -= 1
    elif matrix[i, j, k] == matrix[i - 1, j, k] + gap:
      align1.append(xi)
      align2.append('-')
      align3.append('-')
      i -= 1
    elif matrix[i, j, k] == matrix[i, j, k - 1] + gap:
      align1.append('-')
      align2.append('-')
      align3.append(zk)
      k -= 1

  while j > 0 and k > 0:
    yj = seq2[j - 1]
    zk = seq3[k - 1]
    if matrix[i, j, k] == compare(yj, zk) + matrix[i, j - 1, k - 1]:  # 2d using NeedlemanWunsch's compare
      align1.append('-')
      align2.append(yj)
      align3.append(zk)
      j -= 1
      k -= 1
    elif matrix[i, j, k] == matrix[i, j - 1, k] + gap:
      align1.append('-')
      align2.append(yj)
      align3.append('-')
      j -= 1
    elif matrix[i, j, k] == matrix[i, j, k - 1] + gap:
      align1.append('-')
      align2.append('-')
      align3.append(zk)
      k -= 1

  while i > 0:
    xi = seq1[i - 1]
    align1.append(xi)
    align2.append('-')
    align3.append('-')
    i -= 1

  while j > 0:
    yj = seq2[j - 1]
    align1.append('-')
    align2.append(yj)
    align3.append('-')
    j -= 1

  while k > 0:
    zk = seq3[k - 1]
    align1.append('-')
    align2.append('-')
    align3.append(zk)
    k -= 1
  return align1[::-1], align2[::-1], align3[::-1]

def test_matrix():
  seq1 = [token.value for token in RevAI("../data/CallHome_eval/rev/4074_cut.json").get_token_list() if
          token.spk_id == 0]
  seq2 = [token.value for token in RevAI("../data/CallHome_eval/rev/4074_cut.json").get_token_list() if
          token.spk_id == 1]
  target = [token.value for token in CallHome("../data/CallHome_eval/transcripts/4074.cha").get_token_list()]
  # target = ("I am fish what are you" * 200).split()
  # seq1 = ("I am fish" * 200).split()
  # seq2 = ("what are you" * 200).split()
  align1, align2, align3 = backtrack(target, seq1, seq2, get_scoring_matrix_3d(target, seq1, seq2))
  with open("4074_test.csv", 'w') as file:
    output = csv.writer(file)
    output.writerows([align2, align3, align1])

if __name__ == "__main__":
  # seq1 = [token.value for token in RevAI("../data/CallHome_eval/rev/4074_cut.json").get_token_list() if
  #         token.spk_id == 0]
  # seq2 = [token.value for token in RevAI("../data/CallHome_eval/rev/4074_cut.json").get_token_list() if
  #         token.spk_id == 1]
  # target = [token.value for token in CallHome("../data/CallHome_eval/transcripts/4074.cha").get_token_list()]
  # align = MultiSeqAlign(target, seq1, seq2)
  # print(align.matrix)
  # print(align.matrix.shape)
  # align.compute_matrix()
  # target_align, seq1_align, seq2_align = align.backtrack()
  # print(seq1_align)
  # print(seq2_align)
  # print(target_align)

  print(timeit.Timer(test_matrix).timeit(number=1))
