############################################################################################################
#                                  Author: Flora Medeiros Sauerbronn                                       #
#                                           Date: 09/02/2024                                               #
#      This routine creates and process the dataset of a youtube tutorial for genre music classification   #
#                     using MFCCs the channel namos in Sound of AI, the dataset is from                    #
#                             gtzan-dataset-music-genre-classification in Kaggle                           #
############################################################################################################
import os
import librosa

DATASET_PATH = ""
JSON_PATH = "data.json"
SAMPLE_RATE = 22050
DURATION = 30 #measured in seconds
SAMPLE_PER_TRACK = sr * DURATION

#json_path -> the json file were store the mfccs and the labels
def save_mfcc(dataset_path,json_path,n_mfcc=13,n_fft = 2048, hop_length=512, num_segments=5):
    #dictionary to store data
    data = {
        "mapping" = ["classical","blues"], #the classes
        "mfcc" = [[...],[...],[...]], #training inputs-> vector negments of the audio that is beeibg classify
        "labels" = [0,0,1], #the outputs that we spect
    }
num_samples_per_segment = int(SAMPLE_PER_TRACK / num_segments)

#loop through all the genres
for i, (dirpath,dirnames,filenames) in enumerate(os.walk(dataset_path)):

    #ensure that we're not in the root level
    if dirpath is not dataset_path:
        pass
        #Save the semantic label
        dirpath_components = dirpath.splt("/") # genre/blues => ["genre","blues"]
        semantic_label = dirpath_components[-1]
        data["mapping"].append(semantic_label)
        
        #Go throght all the file sin the current genre folder
        #process files for a specific genre
        for f in filenames:
            #load the audiofile
            file_path = os.path.join(dirpath,f)
            signal,sr = librosa.load(file_path,sr = SAMPLE_RATE)

            #process segments extracting mfcc and storing data
            for s in range(num_segments):
                start_sample = num_samples_per_segment * s
                finish_sample = start_sample + num_samples_per_segment

                mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample], #examing a slice of the signal
                                            sr=sr,
                                            n_fft=n_fft,
                                            n_mfcc=n_mfcc,
                                            hop_length = hop_length)

    
