import nltk

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

    def match_score(self):
        pass

    def get_distance(self, other):
        return nltk.metrics.edit_distance(self.value, other.value)