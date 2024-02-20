############################################################################################################
#                                  Author: Flora Medeiros Sauerbronn                                       #
#                                           Date: 20/02/2024                                               #
#      This routine is the first oficial test of the Convolutional Neural Network using ResNet-18          #
#                          Tooked the base routine form chat GPT #faith                                    #
############################################################################################################


import torch
from torchvision import transforms, datasets

###############################################################################################################
#IMAGE PRE PROCESSING
###############################################################################################################

# Caminhos para os dados
train_data_path = r'D:\SMALL_IMAGES_CNN\train'
val_data_path = r'D:\SMALL_IMAGES_CNN\val'
test_data_path = r'D:\SMALL_IMAGES_CNN\test'

# Transformações de dados
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

#Note: 
#ImageFolder atribui automaticamente rótulos a partir dos nomes das subpastas dentro da pasta principal.


# Carregar os dados de treinamento, validação e teste
train_dataset = datasets.ImageFolder(root=train_data_path, transform=data_transform)
val_dataset = datasets.ImageFolder(root=val_data_path, transform=data_transform)
test_dataset = datasets.ImageFolder(root=test_data_path, transform=data_transform)

#Note: 
#shuffle no DataLoader determina se os dados devem ser embaralhados (misturados) a cada época durante o treinamento.
#No entanto, nos conjuntos de validação (val_loader) e teste (test_loader), não é necessário embaralhar os dados. 
#Isso ocorre porque esses conjuntos são usados apenas para avaliar o desempenho do modelo, e a ordem dos dados não 
#afeta a avaliação.

# Carregar os dados utilizando DataLoader
batch_size = 64
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


###############################################################################################################
#DEFINING ARCHITECHTURE
###############################################################################################################
import torchvision.models as models
import torch.nn as nn

# Carregar a ResNet-18 pré-treinada
resnet18 = models.resnet18(pretrained=True)

# Congelar os parâmetros da ResNet-18 para que não sejam treinados novamente
for param in resnet18.parameters():
    param.requires_grad = False

# Substituir a camada de classificação final para corresponder ao número de classes (2 para binário)
resnet18.fc = nn.Linear(resnet18.fc.in_features, 2)


###############################################################################################################
#TRAINING THE MODEL
###############################################################################################################
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet18.fc.parameters(), lr=0.001)

# Função de treinamento
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Avaliação no conjunto de validação após cada época
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            correct = 0
            total = 0
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            # Imprimir métricas
            print(f'Epoch {epoch+1}/{num_epochs} -> Loss: {val_loss/len(val_loader)} | Accuracy: {100*correct/total}%')

# Treinamento do modelo
train_model(resnet18, train_loader, val_loader, criterion, optimizer, num_epochs=10)


###############################################################################################################
#EVALUATING THE MODEL
###############################################################################################################
# Função de avaliação
def test_model(model, test_loader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        # Imprimir métricas
        print(f'Test Accuracy: {100*correct/total}%')

# Avaliação do modelo no conjunto de teste
test_model(resnet18, test_loader)


###############################################################################################################
#VISUALIZING
#############################################################################################################



import matplotlib.pyplot as plt
import numpy as np

def test_model_with_visualization(model, test_loader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        wrong_predictions = []

        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Armazenar informações sobre previsões incorretas
            wrong_predictions.extend((predicted != labels).nonzero().squeeze().tolist())

        # Imprimir métricas
        print(f'Test Accuracy: {100*correct/total}%')

        # Exibir imagens das previsões incorretas
        for idx in wrong_predictions:
            image, true_label = test_dataset[idx]
            predicted_label = predicted[idx].item()

            # Desnormalizar a imagem antes de exibir (se a normalização foi aplicada)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = std * image.numpy() + mean

            plt.imshow(np.transpose(image, (1, 2, 0)))
            plt.title(f'True Label: {true_label}, Predicted Label: {predicted_label}')
            plt.show()

# Avaliação do modelo com visualização
test_model_with_visualization(resnet18, test_loader)
