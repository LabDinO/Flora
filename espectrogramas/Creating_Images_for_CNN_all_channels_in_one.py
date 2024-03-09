############################################################################################################
#                                  Author: Flora Medeiros Sauerbronn                                       #
#                                           Date: 07/03/2024                                               #
#      This routine creates spectrograms from the audiofiles that Andrea Dalben provided to DeepVoice.     # 
#  All the csv files that has the labels with the calls catalogation were taken from the GitHub of the NGO.#
#                  It creates the spectrograms of positive calls and negative calls                        #
#    All the channels are in one image usin subplot  first line(ch 1 & ch2) second line(ch3 & ch4)         #   
#          In this version the images has no boarders or graphic features (tiles or labels)                #
#The image names-> Channel + Line of the channel in the df(file) + Number of the spectrogram time intervall#
############################################################################################################

# About the labels of Deep Voice
#If we have a positive call in a channel x+1, it does not show a positive call in the other channels
# BUT IT DOES NOT SHOW A NEGATIVE CALL IN THE OTHER. SO WE ARE NOT CREATING SPECTROGRAMS OF THOSE CHANNELS


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


###Function of spectrogram

def plot_spectrogram(Y, sr, hop_length, i, axs, y_axis="linear", title="Spectrogram"):
    # (400/80, 320/80), dpi=80  TESTE COM DPI PARA DELIMITAÇÃO DE PIXELS
    dpi = 100  # You can adjust this value based on your needs
    width, height = 224 / dpi, 224 / dpi
    ax = axs[i // 2, i % 2]  # Calculate the position in the subplot grid
    ax.figure.set_size_inches(width, height)  # Corrected line
    ax.axis('off')
    ax.margins(0, 0)
    ax.set_ylim(10000, 48000)  # Limiting the low frequency

    # Set the desired frequency range
    fmin = 3000  # 3kHz
    fmax = 144000  # 144kHz

    specshow_obj = librosa.display.specshow(Y[i, :, :],
                                            sr=sr,
                                            hop_length=hop_length,
                                            x_axis="time",
                                            y_axis=y_axis,
                                            cmap='jet',
                                            fmin=fmin,
                                            fmax=fmax,
                                            ax=ax)
    
    specshow_obj.set_clim(-60, 10)  # Set color limits for the individual subplot
    ax.set_xticks([])  # Remove x-axis ticks
    ax.set_yticks([])  # Remove y-axis ticks
    #ax.set_title(title)




#Audio parameters

FRAME_SIZE = 1024 #samples
HOP_SIZE = 512 #samples
sr=96000



#CREATING NEGATIVE IMAGES -> negative
def negative_images(file,file_raw, audio_data,sample_rate):

    #Dando um loc no csv que tem as infos de negativo
    ch = file.loc[(file['label'] == 0)]
    #Usando só o que vamos precisar (Os canais, os audios e os labels são iguais)
    ch = ch[['begin_time','end_time','call_length','filename']]  
    #Agora eu tenho uma lista com os dados de cada call sem estar separado por canal. 
    df = ch.groupby(['begin_time',]).first().reset_index() 

    #Agora vamos começar a recortar
    #RETIRAR ESSA LINHA ->  df = ch.loc[(file['channel'] == x +1)] #Pegando cada channel ( é x+1 porque o channel 1 tem indice 0)
    for i in  range (len(df)):  #indo linha por linha em um canal x onde o label é zero (len(df)) -> qtd linhas do canal x+1 que o label é zero
            if df['call_length'].iloc[i] < 2 and i != len(df) - 1: #Se esse intervalo sem cliques for menor que dois segundo e o contador i não for o último índice de df, vá para a próxima linha não analisa essa sessão e vai pra proxima linha
                i = i+1
            t_start = df['begin_time'].iloc[i] # begin time i
            t_stop = df['end_time'].iloc[i] # end time i
            t_start = math.ceil(t_start)  # arredonda t_start pra cima (EVITA PEGAR O RESTO DO CLICK ANTERIOR)
            #Agora temos que ver se o tempo entre t_start e t_stop é par
            t = t_stop - t_start
            n = int(t // 2)  #n é o número de espectrogramas
            if n == 0: #Para ajeitar calls menores que 2 segundos
                n = 1    
            for j in range (n):
                t_stop = t_start + 2  #atualiza t_stop pra pegar o espectrograma de tamanho 2 segundos
                audio_clip = audio_data[:][int(t_start * sr):int(t_stop * sr)]
                #Agora a gente pega e recorta os espectrogramas 
                S_ch_raw1 = librosa.stft(audio_clip, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
                Y_scale = np.abs(S_ch_raw1) ** 2
                Y = librosa.power_to_db(Y_scale)

                #####Plotando a imagem ######
                # Create a 2x2 subplot
                fig, axs = plt.subplots(2, 2, figsize=(10, 8))

                # Plot spectrograms in each subplot
                for l in range(4):
                    plot_spectrogram(Y, sr, HOP_SIZE, l, axs, title=f"Spectrogram {l + 1}")


                # Remove blank spaces around the outer borders
                plt.subplots_adjust(wspace=0, hspace=0)
                plt.margins(0, 0)

                    
                t_start = t_start + 2  #vai passando pro proximo espectrograma até dar n

                output_folder = 'D:/IMAGES_CNN/greyscale_4channels/test/negative'
                    
                # Salve a figura no formato desejado (por exemplo, PNG)
                output_path = os.path.join(output_folder, f'Neg_Img_test_line_{i+1}_N_{j+1}_{ch["filename"].iloc[i]}_{df["begin_time"].iloc[i]}_{df["end_time"].iloc[i]}.png')
                plt.savefig(output_path,bbox_inches='tight', pad_inches=0)
                plt.close()
                print(f"NEGATIVE -> Audio {audio_names[k]} {k}/{len(audio_names)}, Linha {i}/{len(df)} , Espectrograma número {j}/{n}")



#####################
###ANOTTATION FILE###
#####################
test_file =  pd.read_csv('C:/Users/flora/OneDrive/Documentos/MESTRADO_UFSC/rotinas/python/labels_DeepVoice/annotation_test_clicks.csv')

#Lists all the audio files names with no repeat
audio_names = test_file['filename'].unique()
#
#
#
#CRIAR UM LOOP QUE VÁ NA PASTA DOS AUDIOS E LEIA AUDIO POR AUDIO E FIQUE CRIANDO IMAGENS 
#O loop vai ser file por file
#Lists all the audio files names with no repeat


for k in range (len(audio_names)):
     #the audio file
     file = test_file.loc[(test_file['filename'] == audio_names[k])]
     file_raw = f'D:/AUDIOS/test/{audio_names[k]}.wav'
     audio_data, sample_rate = librosa.load(file_raw, sr=None, mono=False)
     negative_images(file, file_raw, audio_data, sample_rate)
     
