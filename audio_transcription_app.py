import streamlit as st
from faster_whisper import WhisperModel
import torch
import torchaudio
from pydub import AudioSegment
import torchaudio.transforms as T
import os
import io

# Load the Silero VAD model globally
# Load the Silero VAD model
vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=True)
(get_speech_timestamps, _, read_audio, _, _) = utils

# Path to the torch hub cache directory
torch_hub_cache_dir = '/root/.cache/torch/hub'

# Correctly unpack the utilities from the tuple
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils


# Initialize the Whisper model
model_size = "large-v3"
model = WhisperModel(model_size, device="cpu", compute_type="int8")

# Function to transcribe audio
def transcribe_audio(audio_file):
    # Save the uploaded file to a temporary file, as WhisperModel expects a file path
    with open("temp_audio_file", "wb") as f:
        f.write(audio_file.getbuffer())
    # Transcribe the audio file using Faster Whisper
    segments, info = model.transcribe("temp_audio_file", beam_size=5)
    # Concatenate all segments to form the full transcription
    transcription = " ".join(segment.text for segment in segments)
    return transcription

def transcribe_large_audio(audio_file_path):
    print("Loading audio...")
    wav = read_audio(audio_file_path, sampling_rate=16000)
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    print("Audio loaded and reshaped if necessary.")

    sr = 16000  # Define the sample rate
    print("Detecting speech segments...")
    speech_timestamps = get_speech_timestamps(wav, vad_model, threshold=0.5, sampling_rate=sr)
    print(f"Found {len(speech_timestamps)} speech segments.")

    full_transcription = ""
    for i, timestamp in enumerate(speech_timestamps, 1):
        print(f"Processing segment {i}/{len(speech_timestamps)}...")
        start_ms, end_ms = int(timestamp['start'] * 1000), int(timestamp['end'] * 1000)
        audio_chunk = AudioSegment.from_raw(io.BytesIO(wav[:, start_ms*sr//1000:end_ms*sr//1000].numpy()), sample_width=2, frame_rate=sr, channels=1)
        chunk_path = "temp_chunk.wav"
        audio_chunk.export(chunk_path, format="wav")
        chunk_transcription = transcribe_audio_chunk(chunk_path)
        full_transcription += chunk_transcription + " "

    print("Transcription complete.")
    return full_transcription.strip()


def transcribe_audio_chunk(chunk_path):
    segments, info = model.transcribe(chunk_path, beam_size=5)
    transcription = " ".join(segment.text for segment in segments)
    return transcription


def needs_splitting(audio_file):
    size_threshold = 5 * 1024 * 1024  # 50MB for example
    return audio_file.size > size_threshold

# Initialize session state for recording status
if 'recording' not in st.session_state:
    st.session_state.recording = False

# Function to handle recording state
def toggle_recording():
    st.session_state.recording = not st.session_state.recording

# Title for the application
st.title('Audio Transcription Application')

# Section for uploading and transcribing audio files
st.header('Upload and Transcribe Audio')
uploaded_file = st.file_uploader("Choose an audio file (WAV or MP3)", type=['wav', 'mp3'])
if uploaded_file is not None:
    st.write("File uploaded successfully. Processing...")
    if needs_splitting(uploaded_file):
        transcription = transcribe_large_audio(uploaded_file)
    else:
        transcription = transcribe_audio(uploaded_file)
    st.text_area("Transcription", transcription, height=250)
    st.download_button(label="Download Transcription", data=transcription, file_name="transcription.txt", mime="text/plain")

# Section for real-time audio recording and transcription
st.header('Real-time Audio Recording and Transcription')
record_button = st.button('Start Recording' if not st.session_state.recording else 'Stop Recording', on_click=toggle_recording)

if st.session_state.recording:
    st.write('Recording...')
else:
    st.write('Click "Start Recording" to begin recording audio.')

# Instructions or footer
st.markdown("""
**Instructions:**
- Upload an audio file to get the transcription.
- Use the real-time transcription feature for live audio recording and transcription. Click "Start Recording" to begin and "Stop Recording" to end.
""")
