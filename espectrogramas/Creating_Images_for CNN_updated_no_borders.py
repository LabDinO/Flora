############################################################################################################
#                                  Author: Flora Medeiros Sauerbronn                                       #
#                                           Date: 23/01/2024                                               #
#      This routine creates spectrograms from the audiofiles that Andrea Dalben provided to DeepVoice.     # 
#  All the csv files that has the labels with the calls catalogation were taken from the GitHub of the NGO.#
#                  It creates the spectrograms of positive calls and negative calls                        #  
#          In this version the images has no boarders or graphic features (tiles or labels)                #
############################################################################################################


#Packages
import wave
import numpy as np
import os
import librosa
import librosa.display
from pydub import AudioSegment
import IPython.display as ipd
from IPython.display import Audio, display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import csv
import math


#Spectrogram Function ONLY IMAGE NO WHITE LAYOUT
def plot_spectrogram(Y, sr, hop_length, y_axis="linear", title="Spectrogram"):
    plt.figure(figsize=(25, 10))
    plt.axis('off')  # Desative os eixos
    plt.margins(0, 0) #define margens pra zero
    librosa.display.specshow(Y,
                             sr=sr,
                             hop_length=hop_length,
                             x_axis="time",
                             y_axis=y_axis,
                             cmap='jet')

    plt.clim(-40, 20)
    plt.tight_layout()


#Audio parameters

FRAME_SIZE = 1024 #samples
HOP_SIZE = 512 #samples
sr=96000

#CREATING NEGATIVE IMAGES -> negative
def negative_images(file,file_raw, audio_data,sample_rate):
     
    for x in range (4): #Loop que vai passar por cada um dos 4 canais
        spectrogram = audio_data[x] # loc no canal do audio file
        ch = file.loc[(file['label'] == 0)]   #Dando um loc no csv que tem as infos
        #Agora vamos começar a recortar
        df = file.loc[(file['channel'] == x +1)]
        for i in  range (len(df)):  #indo linha por linha em um canal x onde o label é zero
                t_start = df['begin_time'].iloc[i] # begin time i
                t_stop = df['end_time'].iloc[i] # end time i
                #Agora temos que ver se o tempo entre t_start e t_stop é par
                t = t_stop - t_start
                n = int(t // 2)  #n é o número de espectrogramas
                if n == 0: #Para ajeitar calls menores que 2 segundos
                    n = 1    
                for j in range (n):
                    t_start = math.floor(t_start) #arredonda t_start pra baixo
                    t_stop = t_start + 2  #atualiza t_stop pra pegar o espectrograma de tamanho 2 segundos
                    audio_clip = audio_data[x][int(t_start * sr):int(t_stop * sr)]
                    #Agora a gente pega e recorta os espectrogramas 
                    S_ch_raw1 = librosa.stft(audio_clip, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
                    Y_scale = np.abs(S_ch_raw1) ** 2
                    Y_log_scale1 = librosa.power_to_db(Y_scale)
                    plot_spectrogram(Y_log_scale1 , sr, HOP_SIZE,y_axis="linear", title="No Dolphins")
                    
                    t_start = t_start + 2  #vai passando pro proximo espectrograma até dar n

                    output_folder = 'D:/IMAGES_CNN/train/negative'
                    
                    # Salve a figura no formato desejado (por exemplo, PNG)
                    output_path = os.path.join(output_folder, f'chan_{x+1}_line_{i+1}_spec_{j+1}_{ch["filename"].iloc[i]}_{df["begin_time"].iloc[i]}_{df["end_time"].iloc[i]}.png')
                    plt.savefig(output_path)
                    plt.close()


#CREATING IMAGES OF CLICKs -> positives

def click_images(file,file_raw,audio_data,sample_rate):
     
    for x in range (4): #Loop que vai passar por cada um dos três canais
        spectrogram = audio_data[x] # loc no canal do audio file
        ch = file.loc[(file['label'] == 1)] #pegando apenas os que tem cliques
        df = ch.loc[(ch['channel'] == x +1)]  #Vendo o canal correspondentes no csv 
        for i in  range (len(df)):  #LOOP PASSA LINHA A LINHA DE CADA CANAL
                t_start = df['begin_time'].iloc[i] # begin time i
                t_stop = df['end_time'].iloc[i] # end time i
            
                #Agora temos que ver se o tempo entre t_start e t_stop é par
                t = t_stop - t_start
                n = int(t // 2)  #n é o número de espectrogramas

                if n == 0: #Para ajeitar calls menores que 2 segundos
                    n = 1

                for j in range (n):
                    t_start = math.floor(t_start) #arredonda t_start pra baixo
                    t_stop = t_start + 2  #atualiza t_stop pra pegar o espectrograma de tamanho 2 segundos
                    audio_clip = audio_data[x][int(t_start * sr):int(t_stop * sr)]
                    #Agora a gente pega e recorta os espectrogramas 
                    S_ch_raw1 = librosa.stft(audio_clip, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
                    Y_scale = np.abs(S_ch_raw1) ** 2
                    Y_log_scale1 = librosa.power_to_db(Y_scale)
                    plot_spectrogram(Y_log_scale1 , sr, HOP_SIZE,y_axis="linear", title="Yes Dolphins")
                    
                    t_start = t_start + 2  #vai passando pro proximo espectrograma até dar n, porque n é a quantidade de espectrogramas que tem no intervalo anotado no csv

                    output_folder = 'D:/IMAGES_CNN/train/positive'
                    # Salve a figura no formato desejado (por exemplo, PNG)
                    output_path = os.path.join(output_folder, f'chan_{x+1}_line_{i+1}_spec_{j+1}_{ch["filename"].iloc[i]}_{df["begin_time"].iloc[i]}_{df["end_time"].iloc[i]}.png')
                    plt.savefig(output_path)
                    plt.close()


#####################
###ANOTTATION FILE###
#####################
test_file =  pd.read_csv('C:/Users/flora/OneDrive/Documentos/MESTRADO_UFSC/rotinas/python/espectrogramas/labels_DeepVoice/annotation_train_clicks.csv')

#Lists all the audio files names with no repeat
audio_names = test_file['filename'].unique()
#
#
#
#CRIAR UM LOOP QUE VÁ NA PASTA DOS AUDIOS E LEIA AUDIO POR AUDIO E FIQUE CRIANDO IMAGENS 
#O loop vai ser file por file


for k in range (len(audio_names)):
     #the audio file
     file = test_file.loc[(test_file['filename'] == audio_names[k])]
     file_raw = f'D:/AUDIOS/train/{audio_names[k]}.wav'
     audio_data, sample_rate = librosa.load(file_raw, sr=None, mono=False)
     #negative_images(file, file_raw, audio_data, sample_rate)
     click_images(file,file_raw,audio_data,sample_rate)

