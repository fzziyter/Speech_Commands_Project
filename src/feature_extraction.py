import librosa
import numpy as np

def extract_mfcc(file_path, n_mfcc=40, max_len=44):
    """
    Transforme un fichier audio en MFCC 2D (n_mfcc x max_len)
    Args:
        file_path : chemin vers le fichier .wav
        n_mfcc : nombre de coefficients MFCC
        max_len : nombre maximal de frames (padding/truncation)
    Returns:
        mfcc : np.array (n_mfcc x max_len)
    """
    audio, sr = librosa.load(file_path, sr=16000)  # resampling à 16kHz
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    
    # Padding ou truncation pour avoir la même taille
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0,0),(0,pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    
    return mfcc