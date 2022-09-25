import json


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
        start_time = int(input_str[stamp_start + 1:stamp_split]) / 1000
        end_time = int(input_str[stamp_split + 1:stamp_end]) / 1000
        return start_time, end_time

    def get_file_annotation(self, start_with_zero: bool = True, allow_overlap = False): # -> list[tuple[str, float, float]]:
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
            for line in file.readlines():
                if line[0] == '*' and not start:
                    speaker = line[1]
                    start = True
                if "" in line and start:
                    line_start_time, line_end_time = CallHome.get_line_time_stamp(line)
                    start = False
                    if start_with_zero:
                        annotation.append((speaker, line_start_time - file_start_time, line_end_time - file_start_time))
        return annotation

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

    def get_annotation(self): # -> list[tuple[str, float, float]]
        """
        Return the annotation (speaker id, start and end time of each line) of a transcript from rev.ai,
        this method assumes that the time will start at the very beginning (0 or close to 0)
        :return: a list that each element is a tuple containing the speaker id (in str), the start and
        end time in float
        """
        annotation = []
        with open("CallHome_eval/rev/4074_cut.json") as file:
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


if __name__ == "__main__":
    # transcript_4093 = CallHome("CallHome_eval/transcripts/4093.cha")
    # print(transcript_4093.get_file_start_time())
    # for line_annote in transcript_4093.get_file_annotation(True):
    #     print(line_annote)
    rev_4074 = RevAI("CallHome_eval/rev/4074_cut.json")
    print(rev_4074.get_annotation())
    

