

class Sequence:
  def __init__():
    pass
  
  
  
  
class Token:
  def __init__(self, word, spk_id, start=None, end=None):
    self.word = word
    self.spk_id = spk_id
    self.start = start
    self.end = end

  def __str__(self):
    res = ""
    res = res + "(" + str(self.word) + "," + str(self.spk_id) + ")"
    return res
  
  def match_score():
    pass
  
  
  
