import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import os

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

# Função para carregar o modelo treinado
def load_model(model_path, model_class, num_classes):
    device = get_default_device()
    model = model_class(3, num_classes)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    return model

# Função para prever a classe de uma imagem
def predict_image(img_path, model, classes):
    # Transformações para as imagens
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Ajusta as imagens para o tamanho correto
        transforms.ToTensor()
    ])
    
    # Abrir e transformar a imagem
    img = Image.open(img_path)
    img_tensor = transform(img)

    img_tensor = to_device(img_tensor.unsqueeze(0), get_default_device())  # Converte a imagem para batch
    model.eval()
    
    # Previsão da imagem
    with torch.no_grad():
        outputs = model(img_tensor)
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

# Carregar o modelo e prever a classe da folha em uma imagem aleatória
if __name__ == '__main__':
    # Caminho do modelo treinado
    model_path = 'model_epoch_3.pth'

    # Definir as classes manualmente, baseado no treinamento anterior
    class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 
                   'Blueberry___healthy', 'Cherry_(including_sour)___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 
                   'Corn_(maize)___healthy', 'Corn_(maize)___Northern_Leaf_Blight', 'Grape___Black_rot', 
                   'Grape___Esca_(Black_Measles)', 'Grape___healthy', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                   'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 
                   'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___healthy', 
                   'Potato___Late_blight', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                   'Strawberry___healthy', 'Strawberry___Leaf_scorch', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 
                   'Tomato___healthy', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 
                   'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_mosaic_virus', 
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus']

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

    # Caminho da imagem aleatória que você deseja testar
    img_path = 'C:/Users/Olá/Downloads/potato leaf 3.webp'  # Substitua pelo caminho da sua imagem aleatória

    # Fazer a previsão
    predicted_class = predict_image(img_path, model, class_names)
    print(f"Classe prevista para a imagem: {predicted_class}")
