import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Função para carregar o dispositivo padrão (GPU ou CPU)
def get_default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

# Função para mover dados para o dispositivo
def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

# Classe para carregar dados no dispositivo
class DeviceDataLoader():
    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device
        
    def __iter__(self):
        for b in self.dataloader:
            yield to_device(b, self.device)
        
    def __len__(self):
        return len(self.dataloader)

# Função para carregar o modelo treinado
def load_model(model_path, model_class, num_classes):
    device = get_default_device()
    model = model_class(3, num_classes)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    return model

# Função para prever a classe de uma imagem
def predict_image(img, model, classes):
    img = to_device(img.unsqueeze(0), get_default_device())  # Converte a imagem para batch
    model.eval()
    outputs = model(img)
    _, preds = torch.max(outputs, dim=1)
    return classes[preds[0].item()]

# Função para criar um bloco de convolução
def ConvBlock(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)

# Carregar o modelo e testar no conjunto de teste com subpastas de classes
if __name__ == '__main__':
    # Caminho do modelo treinado
    model_path = 'C:\\Users\\Olá\\programacao\\nasa\\New Plant Diseases Dataset(Augmented)/model_epoch_3.pth'

    # Diretório de teste com subpastas de classes
    test_dir = "C:\\Users\\Olá\\programacao\\nasa\\New Plant Diseases Dataset(Augmented)\\test"

    # Definir transformações para as imagens
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Ajusta as imagens para o tamanho correto
        transforms.ToTensor()
    ])

    # Carregar o conjunto de teste utilizando ImageFolder
    test_dataset = ImageFolder(test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Definir as classes baseadas nas pastas de treino
    class_names = test_dataset.classes

    # Definir a arquitetura do modelo
    class CNN_NeuralNet(nn.Module):
        def __init__(self, in_channels, num_diseases):
            super().__init__()
            self.conv1 = ConvBlock(in_channels, 64)
            self.conv2 = ConvBlock(64, 128, pool=True) 
            self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))
            self.conv3 = ConvBlock(128, 256, pool=True) 
            self.conv4 = ConvBlock(256, 512, pool=True)
            self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
            self.classifier = nn.Sequential(nn.MaxPool2d(4), nn.Flatten(), nn.Linear(512, num_diseases))
        
        def forward(self, x):
            out = self.conv1(x)
            out = self.conv2(out)
            out = self.res1(out) + out
            out = self.conv3(out)
            out = self.conv4(out)
            out = self.res2(out) + out
            out = self.classifier(out)
            return out

    # Carregar o modelo treinado
    model = load_model(model_path, CNN_NeuralNet, len(class_names))

    # Testar o modelo no conjunto de teste e imprimir resultados para cada imagem
    correct = 0
    total = 0
    model.eval()

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = to_device(images, get_default_device()), to_device(labels, get_default_device())
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Para cada imagem, mostrar o resultado
            for i in range(len(images)):
                predicted_class = class_names[predicted[i]]
                actual_class = class_names[labels[i]]
                print(f"Imagem {i+1}: Previsto = {predicted_class}, Real = {actual_class}")

    print(f"Acurácia no conjunto de teste: {100 * correct / total:.2f}%")
