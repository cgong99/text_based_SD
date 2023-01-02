
'''
speaker struct{
  sequences: (start,end, list_of_words)
}
'''
# sort sequences among all speakers to generate transcripts

from bs4 import BeautifulSoup
import os

class Speaker:
  def __init__(self, name:str) -> None:
    self.utterances = []
    self.name = name
  
  def append_segments(self, start:str, end:str, words:str):
    self.utterances.append((self.name, start, end, words))
  
def get_speakers(file_code):
  speakers = set()
  for file in os.listdir("./Segments"):
    tokens = file.split(".")
    if tokens[0] == file_code:
      speakers.add(tokens[1])
  return list(speakers)

def initSpeaker(file_code):
  # one file
  # file_code + spk ..
  speakers = get_speakers(file_code=file_code)
  initialized_speakers = []
  for spk in speakers:
    # spk = 'A'
    path = 'Segments/' + file_code + '.' + spk + '.' + 'segs.xml' # Segments/Bdb001.A.segs.xml
    word_path = 'Words/' + file_code + '.' + spk + '.' + 'words.xml'
    speaker = Speaker(spk)
    with open(word_path, 'r') as word_file: # open word file
      word_data = word_file.read()
      word_xml = BeautifulSoup(word_data, "xml")
      word_element_list = word_xml.find('root').findChildren()
      with open(path, 'r') as f:
        data = f.read()
        xml_data = BeautifulSoup(data, "xml")
        all_segs = xml_data.find_all('segment')
        utterance_list = []
        for seg in all_segs:
          start = str(round(float(seg['starttime']),2))
          end = str(round(float(seg['endtime']),2))
          id_string = seg.child['href'] # Bdb001.A.words.xml#id(Bdb001.w.902)..id(Bdb001.w.903)
          # id = id_string.split('#')[1][3:-1] # Bdb001.vocalsound.3
          id_raw = id_string.split('#')[1] #id(Bdb001.w.902)..id(Bdb001.w.903)
          if ".." not in id_raw:
            continue
          id_start = id_raw.split("..")[0][3:-1]
          id_end = id_raw.split("..")[1][3:-1]
          # print(id_start, id_end)
          
          utterance = getUtterance(element_list=word_element_list, id_start=id_start, id_end=id_end)
          # print(utterance)
          if len(utterance) > 0:
            speaker.append_segments(start=start, end=end,words=utterance)
          # utterance_list.append((spk, start, end, utterance))
    print("speaker: ", speaker.name)
    print("utterance: ", len(speaker.utterances))
    # print(speaker.utterances[0:5])
    initialized_speakers.append(speaker)
  print("speaker number:", len(initialized_speakers))
  return initialized_speakers
    
def getUtterance(element_list, id_start, id_end):
  start_flag = False
  utterance = ""
  for element in element_list:
    if element["nite:id"] == id_start:
      start_flag = True
      # if element.text:
      #   utterance += element.text
    if start_flag and element.text:
      if element['c'] != "W":
        pass
      elif element.text[0] == "'" and element['c'] != "." and element['c'] != "CM":
        utterance += element.text
      else:
        utterance = utterance + " " + element.text
    if element["nite:id"] == id_end:
      break
  return utterance
      
def sort_all_speaker_utterances(speakers:list):
  utterance_list = []
  for spk in speakers:
    for utterance in spk.utterances:
      utterance_list.append(utterance)
  sorted_utterances = sorted(utterance_list, key = lambda x: float(x[1]))
  # for i in range(10):
  #   print(sorted_utterances[i])
  return sorted_utterances
  
def extract_transcript(file_code):
  utterances = initSpeaker(file_code=file_code)
  sorted_utterances = sort_all_speaker_utterances(utterances) # format: ('D', '13.21', '14.476', "It It it doesn't")
  folder = "extracted_transcripts/"
  output_name = folder + file_code+"_transcript.txt"
  with open(output_name, 'w') as f:
    for utt in sorted_utterances:
      line = utt[0] + "\t" + utt[1] + "-" + utt[2] + "\t" + utt[3]
      f.write(line)
      f.write("\n")
  print(output_name)

if __name__ == "__main__":
  with open('Segments/Bdb001.A.segs.xml', 'r') as f:
    data = f.read()

  Bs_data = BeautifulSoup(data, "xml")
  # segs = Bs_data.find_all('segment')
  # print(segs)
  
  # utterances = initSpeaker('Bdb001')
  # sort_all_speaker_utterances(utterances)
  # file_code = 'Bdb001'
  file_code = 'Bed002'
  extract_transcript(file_code)