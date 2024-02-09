############################################################################################################
#                                  Author: Flora Medeiros Sauerbronn                                       #
#                                           Date: 02/02/2024                                               #
#                    Creating a CNN for audio_classification with video turorial from Valerio Velardo      #
#                                   Youtube channel the sound of AI                                        #
############################################################################################################
from torch import nn
from torchinfo import summary
class CNNetwork(nn.Module):

###########################################################################
    #Define the constructer
    def __init__(self):
        super().__init__()
        #Architecthure
        #4 conv blocks /flatten / linear / softmax/
        self.conv1 = nn.Sequential(
            #The first one is a convolution layer -. Apply non linearity Relu then max pooling
            nn.Conv2d(
                in_channels = 1, #Greyscale images
                out_channels = 16, #16 filters in our convolutional layer
                kernel_size = 3, #usual size in conv layer
                stride = 1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        ),
        self.conv2 = nn.Sequential(
            #The first one is a convolution layer -. Apply non linearity Relu then max pooling
            nn.Conv2d(
                in_channels = 16, #Double
                out_channels = 32, 
                kernel_size = 3, #usual size in conv layer
                stride = 1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            #The first one is a convolution layer -. Apply non linearity Relu then max pooling
            nn.Conv2d(
                in_channels = 32, #Double
                out_channels = 64, #
                kernel_size = 3, #usual size in conv layer
                stride = 1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(
            #The first one is a convolution layer -. Apply non linearity Relu then max pooling
            nn.Conv2d(
                in_channels = 64, #Double
                out_channels = 128, 
                kernel_size = 3, #usual size in conv layer
                stride = 1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )


    #Flatten the multidimentional output of the last convolutional block
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(128 * 5 * 4,10) #Dense layer -> (128 * 5 * 4) -> shape of the data from the last convolutional block
        #The output is the number of classes (Urban sound 10 classes)
        self.softmax = nn.Softmax(dim = 1)


    #Foward method
        


###########################################################################
    #Define the data flow from layers   
    def forward(self,input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        logits = self.linear(x)
        predictions = self.softmax(logits)
        return predictions 
    

#if __name__ == "__main__":
#    cnn = CNNetwork()
#    summary(cnn,(1,64,44)) #1 is the number of channels, 64 is the number of mel bands and 44 is the time acess
