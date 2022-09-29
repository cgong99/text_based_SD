from asyncore import file_dispatcher
import json
import os
from symbol import pass_stmt
from tokenize import String


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
        with open(self.file_name) as file:
            utterance = ""
            for line in file.readlines():
                if line[0] == '*' and not start:
                    speaker = line[1]
                    start = True
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
                elif not start:    
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
        input = input.replace("\t", "").replace("\n", "")
        return CallHome.remove_tag(input)
    
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
    
    def get_utterences_by_spkID(self): 
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
    
    def write_txt_transcripts(self, path: String):
        output = self.get_utterences_by_spkID()
        file_path = path+self.basename+".txt"
        print(file_path)
        file = open(file_path, "w")
        for utterence in output:
            line = utterence[0] + ": " + utterence[1] + "\n"
            file.write(line)

    
if __name__ == "__main__":
    # transcript_4093 = CallHome("CallHome_eval/transcripts/4093.cha")
    # print(transcript_4093.get_file_start_time())
    # for line_annote in transcript_4093.get_file_annotation():
    #     print(line_annote)
    rev_4074 = RevAI("CallHome_eval/rev/4074_cut.json")
    annotation = rev_4074.get_spk_time_token()
    print(len(annotation))
    
    # amazon = open("CallHome_eval/amazon/4074.json")
    # data = json.load(amazon)
    # for i in range(8):
    #     print(data["results"]["items"][i]["alternatives"][0]["content"])
    # print("\n")
    # print(len(data["results"]["speaker_labels"]["segments"][0]["items"]))
    
    amazon_test = Amazon("CallHome_eval/amazon/4074.json")
    # print(amazon_test.get_file_annotation())
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


