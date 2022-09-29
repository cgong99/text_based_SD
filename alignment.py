

class Sequence:
  def __init__():
    pass
  
  
  
  
class Token:
  def __init__(self, value, spk_id, start=None, end=None):
    self.value = value
    self.spk_id = spk_id
    self.start = start
    self.end = end

  def __str__(self):
    res = ""
    res = res + "(" + str(self.value) + "," + str(self.spk_id) + ")"
    return res
  
  def match_score():
    pass
  
  
