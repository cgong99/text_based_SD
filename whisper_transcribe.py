import whisper

model = whisper.load_model("base")
result = model.transcribe("CallHome_eval/cut_audio/4093_cut.mp3")
print(result["text"])