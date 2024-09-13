from transformers import pipeline

cls = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", device=0)

res = cls("test.mp3")

print(res)
