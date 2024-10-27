import whisper
import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
from datetime import datetime
import os  # Tilføj denne import i toppen af filen
from tqdm import tqdm
import time

def optag_lyd(varighed=5, sample_rate=16000):
    """
    Optager lyd fra mikrofonen
    varighed: Optagelsens længde i sekunder
    sample_rate: Sampling rate i Hz
    """
    print(f"Starter optagelse i {varighed} sekunder...")
    
    # Tilføj fremskridtsindikator
    with tqdm(total=varighed, desc="Optager", unit="sek") as pbar:
        optagelse = sd.rec(int(varighed * sample_rate), 
                          samplerate=sample_rate, 
                          channels=1)
        for _ in range(varighed):
            time.sleep(1)
            pbar.update(1)
        sd.wait()
    
    print("Optagelse færdig!")
    
    # Generer filnavn med tidsstempel
    tidsstempel = datetime.now().strftime("%Y%m%d_%H%M%S")
    filnavn = f"optagelse123.wav"
    
    # Gem optagelsen som WAV-fil
    wav.write(filnavn, sample_rate, optagelse)
    return filnavn

def transcribe_audio(audio_file_path, model_name="base"):
    # Indlæs modellen
    model = whisper.load_model(model_name)
    
    # Transskriber lydfilen
    result = model.transcribe(audio_file_path)
    
    # Returner den transskriberede tekst
    return result["text"]

if __name__ == "__main__":
    lydfil = "optagelse123.wav"
    
    # Tjek om filen eksisterer
    if not os.path.exists(lydfil):
        print("Den specificerede lydfil findes ikke. Laver en ny optagelse...")
        varighed = int(input("Indtast optagelsens varighed i sekunder: "))
        lydfil = optag_lyd(varighed=varighed)
    else:
        print(f"Bruger eksisterende lydfil: {lydfil}")
    
    # Transskriber optagelsen
    print("Transskriberer optagelsen...")
    transcribed_text = transcribe_audio(lydfil)
    print("\nTransskription:")
    print(transcribed_text)
