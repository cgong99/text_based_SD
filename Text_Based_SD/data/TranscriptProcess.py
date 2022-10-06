import json
import os
from tokenize import String
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


class CallHome:

    def __init__(self, file_name: str):
        self.file_name = file_name

    def get_file_start_time(self) -> float:
        """
        :return: Start time in the transcript in second
        """
        with open(self.file_name) as file:
            for line in file.readlines():
                if "" in line:
                    first_stamp_start = line.find("")
                    first_stamp_split = line[first_stamp_start:].find('_')
                    return int(line[first_stamp_start + 1:first_stamp_split + first_stamp_start]) / 1000

    def get_file_end_time(self) -> float:
        """
        :return: End time in the transcript in second
        """
        with open(self.file_name) as file:
            for line in reversed(file.readlines()):
                if "" in line:
                    last_stamp_start = line.find("")
                    last_stamp_split = line[last_stamp_start:].find('_')
                    last_stamp_end = line[last_stamp_start + last_stamp_split:].find("")
                    return int(line[
                               last_stamp_start + last_stamp_split + 1:last_stamp_start + last_stamp_split + last_stamp_end]) / 1000

    @staticmethod
    def get_line_time_stamp(input_str: str): # -> tuple[float, float]:
        """

        :param input_str: Each line of the transcript file
        :return: start and end time of that line according to the transcript
        """
        stamp_start = 0
        stamp_split = 0
        stamp_end = 0
        for i in range(len(input_str)):
            if input_str[i] == '' and stamp_start == 0 and stamp_end == 0:
                stamp_start = i
                continue
            if input_str[i] == '_' and stamp_start != 0 and stamp_end == 0:
                stamp_split = i
                continue
            if input_str[i] == '' and stamp_start != 0 and stamp_end == 0:
                stamp_end = i
                continue
        start_time = int(input_str[stamp_start + 1:stamp_split])/1000
        end_time = int(input_str[stamp_split + 1:stamp_end])/1000
        return start_time, end_time
    
    def get_file_annotation(self, start_with_zero: bool = True, allow_overlap = True, with_utterances=False): # -> list[tuple[str, float, float]]:
        """
        Return the annotation (speaker id, start and end time of each line) of a transcript
        :param start_with_zero: boolean that determine if the annotation should begin with 0 second and
        subtract all other time accordingly
        :param file_name:
        :return: a list that each element is a tuple containing the speaker id (in str), the start and
        end time in float
        """
        annotation = []
        file_start_time = self.get_file_start_time()
        start = False
        header = True  # skip the header information to the first line with "*"
        with open(self.file_name) as file:
            utterance = ""
            for line in file.readlines():
                if line[0] == '*' and start and "" not in line :
                    utterance = utterance + " " + line[line.find(":")+1:]
                if line[0] == '*' and not start:
                    speaker = line[1]
                    start = True
                    header = False
                    if "" not in line:
                        utterance = line[line.find(":")+1:]
                if "" in line and start:
                    line_start_time, line_end_time = CallHome.get_line_time_stamp(line)
                    start = False
                    if with_utterances:
                        utterance = utterance + " " + line[line.find(":")+1:line.find("")]
                        utterance = self.clean_utterance(utterance)
                        annotation.append((speaker, line_start_time - file_start_time, line_end_time - file_start_time, utterance))
                        utterance = ""  
                        continue
                    else:
                        annotation.append((speaker, line_start_time - file_start_time, line_end_time - file_start_time))  
                        utterance = ""
                elif line[0] != '*' and not header:
                    utterance = utterance + " " + line[line.find(":")+1:]
        if not allow_overlap:
            annotation = self.solve_overlap(annotation)
        return annotation

    def solve_overlap(self, annotation):
        delete = []
        change = {} # #number: new_tuple
        for i in range(len(annotation)-1):
            for j in range(i+1, len(annotation)):
                intersection = self.compute_intersection_length(annotation[i],annotation[j])
                if intersection > 0:
                    if annotation[j][2] < annotation[i][2]: # annotation2 within annotation1, delete annotation2
                        delete.append(j)
                    else:
                        change[j] = (annotation[j][0],annotation[i][2], annotation[j][2]) #start time change to annotation1's start time
        
        new_annotation = []
        for i in range(len(annotation)):
            if i in delete:
                continue
            elif i in change.keys():
                new_annotation.append(change[i])
            else:
                new_annotation.append(annotation[i])
        return new_annotation

    
    def compute_intersection_length(self, A, B):
        """Compute the intersection length of two tuples.
        Args:
            A: a (speaker, start, end) tuple of type (string, float, float)
            B: a (speaker, start, end) tuple of type (string, float, float)
        Returns:
            a float number of the intersection between `A` and `B`
        """
        max_start = max(A[1], B[1])
        min_end = min(A[2], B[2])
        return max(0.0, min_end - max_start)
    
    def get_token_list(self):
        """

        Returns:
            List: a list of Tokens containing word, spk_id
        """
        tokens = []
        annotations = self.get_file_annotation(with_utterances=True)
        for segment in annotations:
            utterance = segment[3]
            spk_id = segment[0]
            words = utterance.split()
            for word in words:
                tokens.append(Token(word, spk_id))
        return tokens
    
    def get_overlapped_part(self):
        """get utterances that entirely overlapped with other utterance
        """
        annotations = self.get_file_annotation(with_utterances=True)
        for i in range(len(annotations)-1):
            for j in range(i+1, len(annotations)):
                intersection = self.compute_intersection_length(annotations[i],annotations[j])
                if annotations[i][1] < annotations[j][1] and annotations[i][2] > annotations[j][2]:
                    print(annotations[i],"\n", annotations[j])

    
    @staticmethod
    def change_time_stamp(input_str: str) -> str:
        """
        Change the time stamp with unreadable character to the format of "begin_second-end_second"
        :param input_str: line of the file
        :return: A new string with new expression of time stamp
        """
        if "" in input_str:
            stamp_start = 0
            stamp_split = 0
            stamp_end = 0
            for i in range(len(input_str)):
                if input_str[i] == '' and stamp_start == 0 and stamp_end == 0:
                    stamp_start = i
                    continue
                if input_str[i] == '_' and stamp_start != 0 and stamp_end == 0:
                    stamp_split = i
                    continue
                if input_str[i] == '' and stamp_start != 0 and stamp_end == 0:
                    stamp_end = i
                    continue
            start_time = int(input_str[stamp_start + 1:stamp_split]) / 1000
            end_time = int(input_str[stamp_split + 1:stamp_end]) / 1000
            output_str = input_str.replace(input_str[stamp_start:stamp_end + 1], "{}-{}".format(start_time, end_time))
            return output_str
        else:
            return input_str

    def clean_utterance(self, input:str):
        input = CallHome.remove_tag(input)
        return ''.join(e for e in input if e.isalnum() or e == " ")
    
    @staticmethod
    def remove_tag(input_str: str) -> str:
        """
        Remove the tag begin with "&="
        :param input_str: line of the file
        :return: A new string without the tag begin with "&="
        """
        if "&=" in input_str:
            output_str = []
            tag_char = False
            for i in range(len(input_str) - 1):
                if input_str[i] == '&' and input_str[i + 1] == '=':
                    tag_char = True
                if not tag_char:
                    output_str.append(input_str[i])
                if input_str[i] == ' ' and tag_char:
                    tag_char = False
            return ''.join(output_str)
        else:
            return input_str  # make no changes




class RevAI:

    def __init__(self, file_name: str):
        self.file_name = file_name
        self.basename = self.get_file_basename()
        with open(self.file_name) as file:
            self.data = json.load(file)

    def get_file_basename(self):
        with_extension = os.path.basename(self.file_name)
        basename = with_extension[:with_extension.find(".")]
        return basename
    
    def get_file_annotation(self): # -> list[tuple[str, float, float]]
        """
        Return the annotation (speaker id, start and end time of each line) of a transcript from rev.ai,
        this method assumes that the time will start at the very beginning (0 or close to 0)
        :return: a list that each element is a tuple containing the speaker id (in str), the start and
        end time in float
        """
        annotation = []
        with open(self.file_name) as file:
            data = json.load(file)
            for monologue in data["monologues"]:
                speaker_id = monologue["speaker"]
                for element in monologue["elements"]:
                    if "ts" in element:
                        start_time = element["ts"]
                        break
                for element in reversed(monologue["elements"]):
                    if "end_ts" in element:
                        end_time = element["end_ts"]
                        break
                annotation.append((str(speaker_id), start_time, end_time))
        return annotation

    def get_spk_time_token(self):
        # (spk_id, start_time, end_time, token)
        res = []
        segments = self.data["monologues"]
        for segment in segments:
            spk_id = segment["speaker"]
            for token in segment["elements"]:
                if token["type"] != "punct":
                    res.append((spk_id, token["ts"], token["end_ts"], token["value"]))
        return res
    
    def get_token_list(self):
        tokens = []
        annotations = self.get_spk_time_token()
        for token in annotations:
            tokens.append(Token(token[3], token[0], start=token[1], end=token[2]))
        return tokens
    
    def remove_special_token(self):
        pass
    
class Amazon:
    
    def __init__(self, file_name: str):
        self.file_name = file_name
        self.basename = self.get_file_basename()
        with open(self.file_name) as file:
            self.data = json.load(file)
            
    def get_file_basename(self):
        with_extension = os.path.basename(self.file_name)
        basename = with_extension[:with_extension.find(".")]
        return basename
        
    def get_file_annotation(self):
        annotation = []
        for segment in self.data["results"]["speaker_labels"]["segments"]:
            annotation.append((str(segment["speaker_label"]), float(segment["start_time"]), float(segment["end_time"])))
        return annotation
    
    def get_utterances_by_spkID(self): 
        """
            output: [(speaker_id, "utterence"), (), ...]
        """
        output = []
        item_count = 0
        for segment in self.data["results"]["speaker_labels"]["segments"]:
            utterence = ""
            speaker_id = segment["speaker_label"]
            for i in range(len(segment["items"])):
                if self.data["results"]["items"][item_count]["type"] == "punctuation":
                    item_count += 1
                utterence += " "
                utterence += self.data["results"]["items"][item_count]["alternatives"][0]["content"]
                item_count += 1
                
            output.append((speaker_id, utterence))
        return output
    
    def get_token_list(self):
        tokens = []
        utterances = self.get_utterances_by_spkID()
        for utterance in utterances:
            spk_id = utterance[0]
            words = utterance[1].split()
            for word in words:
                tokens.append(Token(word, spk_id))
        return tokens
    
    def write_txt_transcripts(self, path: String):
        output = self.get_utterances_by_spkID()
        file_path = path+self.basename+".txt"
        print(file_path)
        file = open(file_path, "w")
        for utterence in output:
            line = utterence[0] + ": " + utterence[1] + "\n"
            file.write(line)

def whole_string_by_spker(tokens: Token):
  res = ""
  spk_map = {}
  spk = tokens[0].spk_id
  spk_map[spk] = "A"
  res = res + str(spk_map[spk]) + ": "
  for token in tokens:
    if token.spk_id not in spk_map:
        spk_map[token.spk_id] = "B"
    if token.spk_id != spk:
        spk = token.spk_id
        res = res + "\n" + str(spk_map[spk]) + ": "
    res = res + " " + token.value
  return res

def txt_transcripts_for_manual_eval(opened_file, output_file_name):
    output = whole_string_by_spker(opened_file.get_token_list())
    f = open(output_file_name, 'w')
    f.write(output)

if __name__ == "__main__":
    transcript_4093 = CallHome("CallHome_eval/transcripts/4093.cha")
    # tokens = transcript_4093.get_token_list()
    # for token in tokens:
    #     print(token)
    
    # rev_4074 = RevAI("CallHome_eval/rev/4074_cut.json")
    # # annotation = rev_4074.get_spk_time_token()
    # tokens = rev_4074.get_token_list()
    # for token in tokens:
    #     print(token)
    
    # amazon = open("CallHome_eval/amazon/4074.json")
    # data = json.load(amazon)
    # for i in range(8):
    #     print(data["results"]["items"][i]["alternatives"][0]["content"])
    # print("\n")
    # print(len(data["results"]["speaker_labels"]["segments"][0]["items"]))
    
    amazon_test = Amazon("CallHome_eval/amazon/4074.json")
    tokens = amazon_test.get_token_list()
    output = whole_string_by_spker(tokens)
    
    
    # amazon_test.get_utterences_by_spkID()
    # print(amazon_test.get_file_basename())
    # amazon_test.write_txt_transcripts("CallHome_eval/amazon/txt/")
    
    # amazon_path = "CallHome_eval/amazon/"
    # files = os.listdir(amazon_path)
    # for file in files:
    #     file_path = amazon_path + file
    #     if os.path.isdir(file_path):
    #         continue
    #     Amazon(file_path).write_txt_transcripts("CallHome_eval/amazon/txt/")


    callHome_4074 = CallHome("CallHome_eval/transcripts/4074.cha")
    callHome_4074.get_overlapped_part()
    amazon_4074 = Amazon("CallHome_eval/amazon/4074.json")
    rev_4074 = RevAI("CallHome_eval/rev/4074_cut.json")
    print(len(amazon_4074.get_token_list()))
    # for token in rev_4074.get_token_list():
    #     print(token)
    # txt_transcripts_for_manual_eval(callHome_4074, "./4074_ground_truth.txt")
    # txt_transcripts_for_manual_eval(amazon_4074, "./4074_amazon.txt")
    # txt_transcripts_for_manual_eval(rev_4074, "./4074_rev.txt")