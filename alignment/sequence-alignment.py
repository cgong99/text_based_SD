import numpy as np



def LevenshteinDistance(s, t):
  m, n = len(s), len(t)
  d = np.zeros((m, n))

  for i in range(1,m):
    d[i, 0] = i
 
  for j in range(1,n):
    d[0, j] = j
 
  for j in range(1,n):
    for i in range(1,m):
      if s[i] == t[j]:
        substitutionCost = 0
      else:
        substitutionCost = 1
        
      d[i, j] = min(d[i-1, j] + 1, d[i, j-1] + 1, d[i-1, j-1] + substitutionCost) 
  return d[m-1, n-1]

class NW:
  def __init__(self):
    pass
  
  def backtrack(self):
    pass
  
  def compute_table(self):
    pass

if __name__ == "__main__":
  s = "fuck"
  t = "Hello"
  print(LevenshteinDistance(s,t))