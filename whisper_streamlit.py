import streamlit as st
import whisper
import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
from datetime import datetime
import os
from tqdm import tqdm
import time
import queue
import threading

# Initialiser Whisper model i session state
if 'model' not in st.session_state:
    st.session_state.model = whisper.load_model("base")

def optag_lyd_batch(varighed=5, sample_rate=16000):
    """Batch optagelse af lyd"""
    optagelse = sd.rec(int(varighed * sample_rate), 
                      samplerate=sample_rate, 
                      channels=1)
    
    # Vis fremskridtsbar
    progress_bar = st.progress(0)
    for i in range(varighed):
        time.sleep(1)
        progress_bar.progress((i + 1) / varighed)
    
    sd.wait()
    progress_bar.empty()
    
    filnavn = f"optagelse_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    wav.write(filnavn, sample_rate, optagelse)
    return filnavn

def optag_og_transkriber_live(varighed=30, sample_rate=16000, chunk_size=4):
    """Næsten live transkription"""
    tekst_output = st.empty()
    stop_optagelse = threading.Event()
    audio_queue = queue.Queue()
    
    def audio_callback(indata, frames, time, status):
        if status:
            print(status)
        audio_queue.put(indata.copy())
    
    # Start optagelse i baggrunden
    stream = sd.InputStream(callback=audio_callback,
                          channels=1,
                          samplerate=sample_rate)
    
    with stream:
        # Vis fremskridtsbar
        progress_bar = st.progress(0)
        accumulated_text = ""
        
        for i in range(0, varighed, chunk_size):
            if stop_optagelse.is_set():
                break
                
            # Opdater fremskridtsbar
            progress_bar.progress(min(1.0, (i + chunk_size) / varighed))
            
            # Saml lyddata for de næste sekunder
            chunk_data = []
            for _ in range(int(chunk_size * sample_rate / 1024)):
                try:
                    chunk_data.append(audio_queue.get(timeout=1))
                except queue.Empty:
                    break
            
            if chunk_data:
                # Konverter til numpy array og gem midlertidigt
                audio_chunk = np.concatenate(chunk_data)
                temp_filename = "temp_chunk.wav"
                wav.write(temp_filename, sample_rate, audio_chunk)
                
                # Transkriber chunk
                result = st.session_state.model.transcribe(temp_filename)
                accumulated_text += result["text"] + " "
                tekst_output.markdown(f"**Transkription:**\n{accumulated_text}")
                
                # Ryd midlertidig fil
                os.remove(temp_filename)
    
    return accumulated_text

def main():
    st.title("🎤 MAGS' - Whisper Transskription 🎧")
    
    tab1, tab2 = st.tabs(["Batch Optagelse", "Live Transkription"])
    
    with tab1:
        st.header("Batch Optagelse og Transkription")
        varighed = st.slider("Vælg optagelsens varighed (sekunder)", 1, 30, 5, key='batch_duration')
        
        if st.button("Start Batch Optagelse"):
            with st.spinner("Optager..."):
                lydfil = optag_lyd_batch(varighed=varighed)
                st.success("Optagelse færdig!")
                
            with st.spinner("Transskriberer..."):
                resultat = st.session_state.model.transcribe(lydfil)
                st.markdown(f"**Transkription:**\n{resultat['text']}")
    
    with tab2:
        st.header("'Næsten Live' Transkription")
        live_varighed = st.slider("Vælg optagelsens varighed (sekunder)", 10, 120, 30, key='live_duration')
        chunk_size = st.slider("Chunk størrelse (sekunder)", 2, 10, 4, key='chunk_size')
        
        if st.button("Start Live Transkription"):
            optag_og_transkriber_live(varighed=live_varighed, chunk_size=chunk_size)

if __name__ == "__main__":
    main()
