import os
from pydub import AudioSegment


def cut_audio(export_path, audio_path, trancript_path, audio, transcript):
    start, end = find_start_end_time(trancript_path + transcript)
    print(start / 1000, end / 1000)
    sound = AudioSegment.from_mp3(audio_path + audio)
    print(type(sound))
    extract = sound[start:end]

    audio_name = audio[:audio.find(".")]
    export_name = export_path + audio_name + "_cut.mp3"
    print(export_name)
    extract.export(export_name, format="mp3")


def find_start_end_time(transcript_name):
    start_time, end_time = None, None
    with open(transcript_name) as file:
        time_stamp = chr(21)
        lines = file.readlines()
        for line in lines:
            if time_stamp in line:
                start = line.find(str(time_stamp)) + 1
                end = line.find('_')
                start_time = int(line[start:end])
                break

        for line in reversed(lines):
            if time_stamp in line:
                start = line.find('_') + 1
                for i in range(start, len(line)):
                    if line[i] == time_stamp:
                        end_time = int(line[start:i])
                break

    return start_time, end_time


def match_file_without_extension(file1, file2):
    name1 = file1[:file1.find(".")]
    name2 = file2[:file2.find(".")]
    return name1 == name2


if __name__ == "__main__":
    audio_path = "CallHome_eval/raw_audio/"
    transcript_path = "CallHome_eval/transcripts/"
    export_path = "CallHome_eval/cut_audio/"

    all_audio = os.listdir(audio_path)
    all_transcripts = os.listdir(transcript_path)

    for audio_file in all_audio:
        for transcript_file in all_transcripts:
            if match_file_without_extension(audio_file, transcript_file):
                cut_audio(export_path, audio_path, transcript_path, audio_file, transcript_file)

    # cut_audio(audio, file)
