from tokenize import String
import numpy as np
from numpy import ndarray
from typing import List, Tuple

from sklearn.metrics import get_scorer
from NeedlemanWunsch import edit_distance, get_scoring_matrix, compare


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
            
if __name__ == "__main__":
  # target = "AGTTG"
  # seq1 = "AG"
  # seq2 = "GTTG"
  target = "I'm going to you know uhm Georige"
  seq1 = "I'm gonna go to uhm Georige"
  seq2 = "you know"
  # align = MultiSeqAlign(target, seq1, seq2)
  align = MultiSeqAlign(target.split(), seq1.split(), seq2.split())
  print(align.matrix)
  print(align.matrix.shape)
  align.compute_matrix()
  print(align.backtrack())
