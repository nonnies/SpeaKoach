import time
import sounddevice as sd
from scipy.io.wavfile import write
from interviewlogic import client
import os
import streamlit as st
from mutagen.mp3 import MP3


class AudioLOGIC:

    def record_audio(self, filename="input.wav", duration=5, samplerate=44100):
        print("Recording... Speak now")
        audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
        sd.wait()
        write(filename, samplerate, audio)
        print("Recording saved!")
        return filename
        
    def transcribe_audio(self, filepath):
        audio_file = open(filepath, "rb")
        transcript = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe", 
            file=audio_file
        )
        return transcript.text

    def speak_text(self, text):
        if os.path.exists("output.mp3"):
            os.remove("output.mp3")  # Remove old file if it exists
        filename = "output.mp3"
        with client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice="coral",
            input=text) as response:
            response.stream_to_file(filename)

        audio = MP3(filename)
        duration = audio.info.length
        print(f"Duration: {duration:.2f} seconds")

        st.audio(filename, autoplay=True)
        # Inject raw HTML — no controls visible
        st.markdown("""
            <style>
            audio { display: none !important; }
            </style>
        """, unsafe_allow_html=True)
        time.sleep(duration)

