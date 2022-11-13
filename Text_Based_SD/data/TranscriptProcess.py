from curses import newwin
import json
import os
from tokenize import String


class Token:
    def __init__(self, value, spk_id, start: float =None, end: float =None):
        self.value = value
        self.spk_id = spk_id
        self.start = start
        self.end = end

    def __str__(self):
        res = ""
        res = res + "(" + str(self.value) + "," + str(self.spk_id)
        if self.start or self.end:
            res = res + ", " + str(self.start) + ", " + str(self.end)
        res += ")"
        return res

    def match_score(self):
        pass

    def get_distance(self, other):
        pass


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
            List: a list of Tokens containing word, spk_id, with (start_time, end_time) if the first or last token in an utterance
        """
        tokens = []
        annotations = self.get_file_annotation(with_utterances=True)
        for segment in annotations:
            utterance = str.lower(segment[3])
            spk_id = segment[0]
            start, end = segment[1], segment[2]
            words = utterance.split()
            for i in range(len(words)):
                if i == 0:
                    tokens.append(Token(words[i], spk_id, start=start))
                elif i == len(words)-1:
                    tokens.append(Token(words[i], spk_id, end=end))
                else:
                    tokens.append(Token(words[i], spk_id))
                
                
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

    def __init__(self, file_name: str, istxt=False):
        self.file_name = file_name
        self.basename = self.get_file_basename()
        self.istxt = istxt
        if istxt:
            with open(self.file_name) as file:
                self.data = file.readlines()
        else:
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
        if self.istxt:
            print("need to use json file")
            return
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
        if self.istxt:
            print("need to use json file")
            return
        res = []
        segments = self.data["monologues"]
        for segment in segments:
            spk_id = segment["speaker"]
            for token in segment["elements"]:
                if token["type"] != "punct":
                    res.append((spk_id, token["ts"], token["end_ts"], token["value"]))
        return res

    def get_utterances_by_spkID(self):
        res = []
        for monologue in self.data["monologues"]:
            spk_id = monologue["speaker"]
            utterance = ""
            for element in monologue["elements"]:
                if element["type"] != "punct":
                    # if element["value"][0] == "<":
                    #     continue
                    utterance = utterance + " " + element["value"]
            res.append((spk_id, utterance))
        return res
    
    def get_token_list(self):
        if self.istxt:
            tokens = []
            for line in self.data:
                arr = line.split(" ")
                spk = arr[1]
                utt = arr[9:-2]
                for word in utt:
                    if word[0] == "<":
                        continue
                    new_word = "".join(e for e in word if e.isalnum())
                    new_word = new_word.lower()
                    tokens.append(Token(new_word, spk))
            return tokens
        else:
            tokens = []
            annotations = self.get_spk_time_token()
            for token in annotations:
                word = str.lower(token[3])
                tokens.append(Token(word, token[0], start=token[1], end=token[2]))
            return tokens
    
    def remove_special_token(self):
        pass
    
class Amazon:
    
    def __init__(self, file_name: str, istxt=False):
        self.file_name = file_name
        self.basename = self.get_file_basename()
        self.istxt = istxt
        if istxt:
            with open(self.file_name) as file:
                self.data = file.readlines()
        else:
            with open(self.file_name) as file:
                self.data = json.load(file)
            
    def get_file_basename(self):
        with_extension = os.path.basename(self.file_name)
        basename = with_extension[:with_extension.find(".")]
        return basename
        
    def get_file_annotation(self):
        if self.istxt:
            print("get annoataion need to be json file")
            return
        annotation = []
        for segment in self.data["results"]["speaker_labels"]["segments"]:
            annotation.append((str(segment["speaker_label"]), float(segment["start_time"]), float(segment["end_time"])))
        return annotation
    
    def get_utterances_by_spkID(self): 
        """
            output: [(speaker_id, "utterence"), (), ...]
        """
        if self.istxt:
            print("need to be json file")
            return
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
        if self.istxt:
            pass
        else:
            tokens = []
            item_count = 0
            for segment in self.data["results"]["speaker_labels"]["segments"]:
                speaker_id = segment["speaker_label"]
                for i in range(len(segment["items"])):
                    if self.data["results"]["items"][item_count]["type"] == "punctuation":
                        item_count += 1
                    word = str.lower(self.data["results"]["items"][item_count]["alternatives"][0]["content"])
                    start = self.data["results"]["items"][item_count]["start_time"]
                    end = self.data["results"]["items"][item_count]["end_time"]
                    tokens.append(Token(word, speaker_id, start=float(start), end=float(end)))  
                    item_count += 1
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

def segment_token_lists(gt: list[Token], output: list[Token]) -> tuple[list[list], list[list]]: 
    """segment ground truth tokens and output tokens using time stamp.

    Args:
        gt (list[Token]): grond truth tokens from get_token_list()
        output (list[Token]): transcriber tokens from get_token_list()

    Returns:
        list[segment1[token..], segment2[token...]....], list[segment1[token..], segment2[token...]....]
    """
    time_window = 100 # make a segment every about 100 seconds
    gt_res = []
    output_res = []
    cuts = []
    end = 0
    # while end < gt[-1].end - time_window:
    for token in gt:
        if token.end and token.end - end > time_window:
            if token.end > gt[-1].end - time_window:
                break
            end = token.end
            cuts.append(end)
    print("cut points: ", cuts)
    gt_count, output_count = 0, 0 
    ### Test
    # append_count = 0
    for cut_point in cuts:
        gt_tokens, output_tokens = [], []
        while not gt[gt_count].end or gt[gt_count].end <= cut_point:
            # if append_count < 1:
                # print(gt[gt_count])
            gt_tokens.append(gt[gt_count])
            gt_count += 1
            if gt[gt_count].end == cut_point: break
        while output[output_count].end <= cut_point + 0.5: # 0.5s tolerance 
            output_tokens.append(output[output_count])
            output_count += 1
    
        gt_res.append(gt_tokens)
        output_res.append(output_tokens)
        # append_count = 1
        
    gt_tokens = []
    while gt_count < len(gt):
        gt_tokens.append(gt[gt_count])
        gt_count += 1
    gt_res.append(gt_tokens)
    output_tokens = []
    while output_count < len(output):
        output_tokens.append(output[output_count])
        output_count += 1
    output_res.append(output_tokens)
    return gt_res, output_res
              
        
    
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


    # callHome_4074 = CallHome("CallHome_eval/transcripts/4074.cha")
    # # callHome_4074.get_overlapped_part()
    # amazon_4074 = Amazon("CallHome_eval/amazon/4074.json")
    # rev_4074 = RevAI("CallHome_eval/rev/4074_cut.json")
    # gt = callHome_4074.get_token_list()
    # amazon = amazon_4074.get_token_list()
    # rev = rev_4074.get_token_list()
    # print(gt[-10:])
    # for token in gt[-30:]:
    #     print(token)
    # for token in rev[-30:]:
    #     print(token)
    # print(len(amazon))

    # gt_seg, output_seg = segment_token_lists(gt, amazon)
    # sum = 0
    # for l in output_seg:
    #     sum += len(l)
    #     print(len(l))
    # print(sum)

    # txt_transcripts_for_manual_eval(callHome_4074, "./4074_ground_truth.txt")
    # txt_transcripts_for_manual_eval(amazon_4074, "./4074_amazon.txt")
    # txt_transcripts_for_manual_eval(rev_4074, "./4074_rev.txt")
    
    rev = RevAI("CallHome_eval/rev/txt/4074_cut.txt",istxt=True)
    for token in rev.get_token_list():
        print(token)