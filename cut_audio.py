from pydub import AudioSegment

def cut_audio(audio, transcript):
  start, end = find_start_end_time(transcript)
  print(start/1000, end/1000)
  sound = AudioSegment.from_mp3(audio)
  print(type(sound))
  extract = sound[start:end]
  
  audio_name = audio[:audio.find(".")]
  export_name = audio_name + "_cut.mp3"
  print(export_name)
  extract.export(export_name, format="mp3")
  pass

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
      
      

if __name__ == "__main__":
  audio = "CallHome_eval/4093.mp3"
  file = "CallHome_eval/4093.cha"
  cut_audio(audio, file)


        