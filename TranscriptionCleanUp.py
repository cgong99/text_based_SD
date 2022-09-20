def get_start_end_time(file_name: str) -> tuple[int, int]:
    """
    Get the start and end time of the transcript in the relative audio file
    :param file_name:
    :return: two int, the first one is the start time in ms, the second one is the end time in ms
    """
    start_time = 0
    end_time = 0
    with open(file_name) as file:
        for line in file.readlines():
            if "" in line:
                first_stamp_start = line.find("")
                first_stamp_split = line[first_stamp_start:].find('_')
                start_time = int(line[first_stamp_start + 1:first_stamp_split + first_stamp_start])
                break

    with open(file_name) as file:
        for line in reversed(file.readlines()):
            if "" in line:
                last_stamp_start = line.find("")
                last_stamp_split = line[last_stamp_start:].find('_')
                last_stamp_end = line[last_stamp_start + last_stamp_split:].find("")
                end_time = int(
                    line[last_stamp_start + last_stamp_split + 1:last_stamp_start + last_stamp_split + last_stamp_end])
                break
    return start_time, end_time


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


if __name__ == "__main__":
    print(get_start_end_time("CallHome_eval/4093.cha"))
