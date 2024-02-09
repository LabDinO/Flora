############################################################################################################
#                                  Author: Flora Medeiros Sauerbronn                                       #
#                                           Date: 02/02/2024                                               #
#                    Testing MNIST dataset with video turorial from Valerio Velardo                        #
#                                   Youtube channel the sound of AI                                        #
############################################################################################################

#Importing
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

#Steps
# 1- donwload the dataset
# 2 - create data loader
# 3 - build a model
# 4 - train
# 5 - save trained model

#Defining constants: 

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = .001




class FeedForwardNet(nn.Module):
    #Constructer defining the layers
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten() #flatten
        self.dense_layers = nn.Sequential(
            nn.Linear(28*28,256),#simple dense layer, images 28*28 pixels
            nn.ReLU(), #activation function
            nn.Linear(256,10) #10 is the number of classes of MNIST
        )#pack togueter multiple layers, data floows sequentionally
        self.softmax = nn.Softmax(dim=1) #Take all the values form the 10 classes and transform where the sum of this 10 will be equal to one
        
        #Specyfing the data flow

    def forward(self,input_data):#indicates pytorch how they manipulate the data in what sequence
        flatten_data = self.flatten(input_data) #pass input data to the flatten layer 
        logits = self.dense_layers(flatten_data)
        predictions = self.softmax(logits)
        return predictions

def download_mnist_datasets():
    train_data = datasets.MNIST(
        root = "data", #where to store the data
        download = True, #plese do donwloaded
        train = True, #the trainsetpart
        transform = ToTensor() #alowus to apply some transformation direct in to our dataser#take a image in and reshape to a new tensor, each value is normalized between zero and one 
    )
    validation_data = datasets.MNIST(
        root = "data", #where to store the data
        download = True, #plese do donwloaded
        train = False, #the trainsetpart
        transform = ToTensor() #alowus to apply some transformation direct in to our dataser#take a image in and reshape to a new tensor, each value is normalized between zero and one 
    )
    return train_data, validation_data

############################################################################
####Creating two more functions ###
def train_one_epoch(model,data_loader,loss_fn,optimiser,device):
    #create a loop for all the samples in the datasets
    #In each interation we will get a batch of samples (the dataloader came handy)
    for inputs, targets in data_loader:
        inputs,targets = inputs.to(device),targets.to(device)

        #calculates loss
        #At each batch we want to calculate the loss 
        #How we calculate the loss ? Well firt we need to get our predictions of the model
        prediction = model(inputs)
        loss = loss_fn(prediction,targets)

        #Back propagate loss and use gradient descent to update weights
        optimiser.zero_grad()#at every interation the optimiser will calculat gradients (decide how to update the weights)
        #But those gradients are saved in each interation, we want that in each tranning interation  at each batch we want to reset the gradients to zero
        #so we can start from scrach
        loss.backward()
        optimiser.step()

    print(f"Loss: {loss.item()}") #printing the loss of the last batch that we had




def train(model,data_loader,loss_fn,optimiser,device,epochs):
    for i in range(epochs):
        print(f"Epoch {i +1}")
        train_one_epoch(model,data_loader,loss_fn,optimiser,device)
        print("------------------------")
    print("Training is done !")





############################################################################
    #DOWNLOAD THE DATA SET
############################################################################
if __name__ == "__main__":
    train_data, _ = download_mnist_datasets()
    print("MNIST dataset donwloaded")

############################################################################
    #CREATING A DATA LOADER
############################################################################

#Its a class that we can use to wrap a dataser, to fech data and load data in batches
#allow uss to donwload heavy datasets
    train_data_loader = DataLoader(train_data,batch_size=BATCH_SIZE)    

############################################################################
    #BUILD A MODEL
############################################################################
    
    #assing our fedfoward model to a device, what is a device ?
    #One is Cuda the other one is CPU. If you are using GPU aceleration the device will be equal to Cuda, otherwize will be CPU
    #Check the acelaration available
    #checking availability
    if torch.cuda.is_available():
        device ="cuda"
    else:
        device = "cpu"
    print(f"Using {device} device in the model!")



    feed_forward_net = FeedForwardNet().to(device)#instantiate

#instantiate loss function + optimiser
loss_fn =nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(feed_forward_net.parameters(),
                             lr = LEARNING_RATE) #We need to stablish the parameters that out optimiser should optimiser, It need to be the same off feeed foward net

############################################################################
    #TRAIN MODEL
############################################################################
train(feed_forward_net,train_data_loader,loss_fn, optimiser,device, EPOCHS)


############################################################################
    #SAVE MODEL
############################################################################

torch.save(feed_forward_net.state_dict(),"feedforwardnet.pth")
#state dict -> a python dictionary that have all the state, the important information regarding layers and parameters that have been trained 
print("Model Ttrained and stored at feedforwardnet.pth")