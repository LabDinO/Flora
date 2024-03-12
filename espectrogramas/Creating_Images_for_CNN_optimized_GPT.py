import os
import math
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd


#Spectrogram Function ONLY IMAGE NO WHITE LAYOUT
def plot_spectrogram(Y, sr, hop_length, y_axis="linear", title="Spectrogram"):
    #(400/80, 320/80), dpi=80  TESTE COM DPI PARA DELIMITAÇÃO DE PIXELS
    dpi = 100  # You can adjust this value based on your needs
    width, height = 224 / dpi, 224 / dpi
    plt.figure(figsize=(width, height), dpi=dpi)
    plt.axis('off')  # Desative os eixos
    plt.margins(0, 0) #define margens pra zero
    plt.ylim(15000, 48000) #Limiiting the low frequency

    # Set the desired frequency range (Deep Convolutional Neural Networks for Detecting Dolphin Echolocation Clicks)
    fmin = 15000  # 3kHz
    fmax = 48000  # 144kH
    librosa.display.specshow(Y,
                             sr=sr,
                             hop_length=hop_length,
                             x_axis="time",
                             y_axis=y_axis,
                             cmap='gray',
                             fmin=fmin,
                             fmax=fmax)

    plt.clim(-60, 10) #Definido por bibliografia (Deep Convolutional Neural Networks for Detecting Dolphin Echolocation Clicks)
    plt.tight_layout()



# Função para criar e salvar espectrogramas
def create_spectrogram(row, label_folder, audio_data, sr):
    t_start = row['begin_time']
    t_stop = math.ceil(row['end_time'])
    t = t_stop - t_start
    n = max(int(t // 2), 1)

    for j in range(n):
        t_stop = t_start + 2
        audio_clip = audio_data[row['channel'] - 1][int(t_start * sr):int(t_stop * sr)]
        S_ch_raw1 = librosa.stft(audio_clip, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
        Y_scale = np.abs(S_ch_raw1) ** 2
        Y_log_scale1 = librosa.power_to_db(Y_scale)
        plot_spectrogram(Y_log_scale1, sr, HOP_SIZE, y_axis="linear", title="Spectrogram")

        output_path = os.path.join(label_folder, f'chan_{row["channel"]}_line_{row.name}_spec_{j + 1}_{row["filename"]}_{row["begin_time"]}_{row["end_time"]}.png')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"{label_folder.upper()} -> Audio {audio_names[k]} {k}/{len(audio_names)}, Canal {row['channel']}/4, Linha {row.name}/{len(row)}, Espectrograma número {j + 1}/{n}")

# Função para criar espectrogramas para um DataFrame
def create_spectrograms(df, label_folder, audio_data, sr):
    df.apply(lambda row: create_spectrogram(row, label_folder, audio_data, sr), axis=1)

# Parâmetros do áudio
FRAME_SIZE = 1024  # samples
HOP_SIZE = 512  # samples

# Leitura do arquivo de anotações
test_file = pd.read_csv('C:/Users/flora/OneDrive/Documentos/MESTRADO_UFSC/rotinas/python/labels_DeepVoice/annotation_test_clicks.csv')

# Listar todos os nomes dos arquivos de áudio sem repetição
audio_names = test_file['filename'].unique()

# Loop através dos arquivos de áudio
for k in range(len(audio_names)):
    # Carregando os dados do áudio
    file = test_file.loc[(test_file['filename'] == audio_names[k])]
    file_raw = f'D:/AUDIOS/test/{audio_names[k]}.wav'
    audio_data, sample_rate = librosa.load(file_raw, sr=None, mono=False)

    # Criar espectrogramas para chamadas negativas
    negative_df = file.loc[file['label'] == 0]
    create_spectrograms(negative_df, 'D:/IMAGES_CNN/colorful_rainbow/test/negative', audio_data, sample_rate)

    # Criar espectrogramas para chamadas positivas
    positive_df = file.loc[file['label'] == 1]
    create_spectrograms(positive_df, 'D:/IMAGES_CNN/colorful_rainbow/test/positive', audio_data, sample_rate)
