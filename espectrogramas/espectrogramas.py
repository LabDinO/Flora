import wave
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import spectrogram

def plot_spectrogram(ax, channel_data, sample_rate):
    f, t, Sxx = spectrogram(channel_data, fs=sample_rate)

    im = ax.pcolormesh(t, f, 10 * np.log10(Sxx), shading='auto')
    return im
def update(frame, audio_data, num_frames, num_channels, sample_rate, axes):
    num_frames_per_plot = 100  # Number of frames per plot

    if (frame + 1) * num_channels * num_frames_per_plot > num_frames:
        frame_end = num_frames
    else:
        frame_end = (frame + 1) * num_channels * num_frames_per_plot

    frame_start = frame * num_channels * num_frames_per_plot

    for channel_index in range(num_channels):
        ax = axes[channel_index]
        channel_data = audio_data[frame_start + channel_index::num_channels]
        
        im = plot_spectrogram(ax, channel_data, sample_rate)
        ax.set_title(f'Frame {frame + 1} - Canal {channel_index + 1}')
        ax.set_ylabel('Frequência [Hz]')
        ax.set_xlabel('Tempo [s]')
def main():
    audio_file_path = 'C:/Users/flora/OneDrive/Documentos/MESTRADO_UFSC/Deep_Voice/audio_files/PAM_MF_20181228_090000_000.wav'

    with wave.open(audio_file_path, 'rb') as wav_file:
        sample_rate = wav_file.getframerate()
        num_channels = wav_file.getnchannels()
        num_frames = wav_file.getnframes()

        audio_data = wav_file.readframes(num_frames)
        audio_data = np.frombuffer(audio_data, dtype=np.int16)

    #num_frames_per_plot = 100  # Number of frames per plot
    total_frames = num_frames // (num_channels * 100)

    fig, axes = plt.subplots(nrows=num_channels, ncols=1, figsize=(10, 8))
    plt.tight_layout()

    ani = FuncAnimation(fig, update, frames=total_frames, fargs=(audio_data, num_frames, num_channels, sample_rate, axes))

    plt.show()

if __name__ == '__main__':
    main()
