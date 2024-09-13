import gradio as gr
from transformers import pipeline

# Initialize the ASR pipeline
cls = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", device=0)


def transcribe_audio(audio):
    # The audio file is a tuple containing the path and sample rate
    audio_path = audio[0] if isinstance(audio, tuple) else audio

    # Perform speech recognition
    result = cls(audio_path)

    # Return the transcribed text
    return result["text"]


# Create the Gradio interface
iface = gr.Interface(
    fn=transcribe_audio,
    inputs=gr.Audio(type="filepath"),
    outputs="text",
    title="Audio to Speech Transcription",
    description="Upload an audio file to transcribe it to text.",
)

# Launch the interface
iface.launch()
