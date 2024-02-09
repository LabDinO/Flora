import librosa, librosa.display
import matplotlib.pyplot as plt

#30 seconds
file = 'blues.00000.wav'

#waveform
signal,sr = librosa.load(file,sr = 22050) # sr * T -> 22050 * 30

#visualizing this waveform

#librosa.display.waveplot(signal,sr=sr)
librosa.display.waveshow(signal, sr=sr,max_points=11025)  # Adjust 'steps' as needed
#plt.xlabel("Time")
#plt.ylabel("Amplitude")
#plt.show()

#From the time domain (wave form) to the frequency domain
#perform a Fast fourier transform
#fft - > spectrum
import numpy as np
fft = np.fft.fft(signal)
magnitude = np.abs(fft)
frequency = np.linspace(0,sr,len(magnitude))

left_frequency = frequency[:int(len(frequency)/2)]
left_magnitude = magnitude[:int(len(magnitude)/2)]

#plt.plot(left_frequency,left_magnitude) #static snap shot of the hole sound, we whant to understand how these frequencys are contributing to the ovel all sound all time
#plt.xlabel("Frequency")
#plt.ylabel("Magnitude")
#plt.show()
#moast energy are in the lower frequencys
#The plot is simetrycal
#half plot = half of the sample rate
 
# stft -> spectrogram
#give us information of the amplitute as a function of both frequency and time
#number of samples per fft
n_fft = 2048
hop_length = 512
stft = librosa.core.stft(signal, hop_length=hop_length, n_fft=n_fft)

spectrogram = np.abs(stft)
log_spectrogram = librosa.amplitude_to_db(spectrogram)

#librosa.display.specshow(log_spectrogram,sr = sr,hop_length=hop_length)
#plt.xlabel("Time")
#plt.ylabel("Frequency")
#plt.colorbar()#Amplitude
#plt.show()

#MFCCs
MFFCs = librosa.feature.mfcc(signal,n_fft=n_fft,hop_length = hop_length,n_mfcc = 13)
#perfoms the short time fourier
librosa.display.specshow(MFFCs,sr = sr,hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("MFCC")
plt.colorbar()#Amplitude
plt.show()