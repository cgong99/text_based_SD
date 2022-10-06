from tokenize import String
import numpy as np
from numpy import ndarray
from typing import List, Tuple

from sklearn.metrics import get_scorer
from NeedlemanWunsch import edit_distance, get_scoring_matrix


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
      table = get_scoring_matrix(seq1=self.target, seq2=self.seq1)
    elif axes == 'yz':
      y = self.n
      z = self.d
      table = get_scoring_matrix(seq1=self.seq1, seq2=self.seq2)
    elif axes == 'xz':
      x = self.m
      z = self.d
      table = get_scoring_matrix(seq1=self.target, seq2=self.seq2)
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
    pass        
            
if __name__ == "__main__":
  target = "AGTTG"
  seq1 = "AG"
  seq2 = "GTT"
  align = MultiSeqAlign(target, seq1, seq2)
  print(align.matrix)
  print(align.matrix.shape)