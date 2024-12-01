import os
import librosa
import numpy as np
import pandas as pd

# Função para extrair características de um arquivo de áudio
def extract_audio_features(file_path, duration=4, sample_rate=44100):
    # Carregar o arquivo de áudio e ajustar a taxa de amostragem
    signal, sr = librosa.load(file_path, sr=sample_rate)

    # Garantir que o áudio tenha o comprimento especificado (com preenchimento zero, se necessário)
    required_length = int(sample_rate * duration)
    signal = librosa.util.fix_length(signal, size=required_length)

    # Normalizar os valores de amplitude do sinal
    signal = librosa.util.normalize(signal)

    # Configurar parâmetros para extração de características
    hop = 512
    onset_env = librosa.onset.onset_strength(y=signal, sr=sr, hop_length=hop)

    # Extrair características espectrais
    spectral_features = {
        'chroma_stft': np.mean(librosa.feature.chroma_stft(y=signal, sr=sr)),
        'chroma_cqt': np.mean(librosa.feature.chroma_cqt(y=signal, sr=sr)),
        'chroma_cens': np.mean(librosa.feature.chroma_cens(y=signal, sr=sr)),
        'melspectrogram': np.mean(librosa.feature.melspectrogram(y=signal, sr=sr)),
        'rms_energy': np.mean(librosa.feature.rms(y=signal)),
        'spectral_centroid': np.mean(librosa.feature.spectral_centroid(y=signal, sr=sr)),
        'spectral_bandwidth': np.mean(librosa.feature.spectral_bandwidth(y=signal, sr=sr)),
        'spectral_contrast': np.mean(librosa.feature.spectral_contrast(y=signal, sr=sr)),
        'spectral_flatness': np.mean(librosa.feature.spectral_flatness(y=signal)),
        'spectral_rolloff': np.mean(librosa.feature.spectral_rolloff(y=signal, sr=sr)),
        'zero_crossing_rate': np.mean(librosa.feature.zero_crossing_rate(signal)),
        'tempogram': np.mean(librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=hop)),
        'fourier_tempogram': np.mean(librosa.feature.fourier_tempogram(onset_envelope=onset_env, sr=sr, hop_length=hop)),
    }

    # Adicionar coeficientes MFCC
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)
    for i in range(40):
        spectral_features[f'mfcc_{i+1}'] = np.mean(mfccs[i])

    return spectral_features

# Processar uma pasta com arquivos de áudio e extrair características
def process_audio_directory(directory_path):
    for subfolder in os.listdir(directory_path):
        features_data = []
        labels = []
        subfolder_path = os.path.join(directory_path, subfolder)
        if os.path.isdir(subfolder_path):
            for file_name in os.listdir(subfolder_path):
                full_path = os.path.join(subfolder_path, file_name)
                if file_name.endswith('.wav'):
                    labels.append(file_name)
                    features_data.append(extract_audio_features(full_path))

        # Salvar os dados extraídos em um arquivo CSV
        df = pd.DataFrame(features_data)
        df['File_Label'] = labels
        csv_name = f'features_{subfolder}.csv'
        df.to_csv(csv_name, index=False)
        print(f'Arquivo {csv_name} gerado com sucesso.')
        print(df.head())

def main():
    dataset_path = "C:/Users/estef/OneDrive/3oano/ac2/trabalho/UrbanSound8K/audio"
    process_audio_directory(dataset_path)

if __name__ == '__main__':
    main()

