import warnings
warnings.filterwarnings('ignore') 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
import time

# Diretórios de treinamento, validação e teste
Root_dir = "C:\\Users\\Olá\\programacao\\nasa\\New Plant Diseases Dataset(Augmented)"
train_dir = Root_dir + "/train"
valid_dir = Root_dir + "/valid"
test_dir = Root_dir + "/test"

# Definir transformações para data augmentation
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(256),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

# Transformações para validação e teste
test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
])

# Carregar datasets com as transformações correspondentes
train = ImageFolder(train_dir, transform=train_transforms)
valid = ImageFolder(valid_dir, transform=test_transforms)
test = ImageFolder(test_dir, transform=test_transforms)

batch_size = 32

# DataLoaders para treinamento, validação e teste
train_dataloader = DataLoader(train, batch_size, shuffle=True, num_workers=2, pin_memory=True)
valid_dataloader = DataLoader(valid, batch_size, num_workers=2, pin_memory=True)
test_dataloader = DataLoader(test, batch_size, num_workers=2, pin_memory=True)

# Funções para dispositivo
def get_default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
    
def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device
        
    def __iter__(self):
        for b in self.dataloader:
            yield to_device(b, self.device)
        
    def __len__(self):
        return len(self.dataloader)
    
device = get_default_device()

# Movendo data loaders para o dispositivo
train_dataloader = DeviceDataLoader(train_dataloader, device)
valid_dataloader = DeviceDataLoader(valid_dataloader, device)
test_dataloader = DeviceDataLoader(test_dataloader, device)

# Função de acurácia
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):
    
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  
        loss = F.cross_entropy(out, labels)
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print(f"Epoch [{epoch+1}], train_loss: {result['train_loss']:.4f}, "
              f"val_loss: {result['val_loss']:.4f}, val_acc: {result['val_acc']:.4f}, "
              f"test_loss: {result['test_loss']:.4f}, test_acc: {result['test_acc']:.4f}")
        
# Bloco de convolução com BatchNormalization
def ConvBlock(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
             nn.BatchNorm2d(out_channels),
             nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)

# Arquitetura do modelo
class CNN_NeuralNet(ImageClassificationBase):
    def __init__(self, in_channels, num_diseases):
        super().__init__()
        
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True) 
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))
        
        self.conv3 = ConvBlock(128, 256, pool=True) 
        self.conv4 = ConvBlock(256, 512, pool=True)
        
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                       nn.Flatten(),
                                       nn.Linear(512, num_diseases))
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out  
    
model = to_device(CNN_NeuralNet(3, len(train.classes)), device)

@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def fit_OneCycle(epochs, max_lr, model, train_loader, val_loader, test_loader, weight_decay=0,
                 grad_clip=None, opt_func=torch.optim.Adam):
    torch.cuda.empty_cache()
    history = []
    
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_loader))
    
    best_test_acc = 0
    epochs_no_improve = 0
    n_epochs_stop = 5  # Número de épocas sem melhoria para parar
    
    for epoch in range(epochs):
        model.train()
        train_losses = []
        lrs = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()

            if grad_clip: 
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
                
            optimizer.step()
            optimizer.zero_grad()
            lrs.append(get_lr(optimizer))
            sched.step()

        # Avaliar no conjunto de validação e teste
        result = evaluate(model, val_loader)
        test_result = evaluate(model, test_loader)
        result['test_loss'] = test_result['val_loss']
        result['test_acc'] = test_result['val_acc']
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result)
        history.append(result)
        
        # Verificar melhoria na acurácia do teste
        if result['test_acc'] > best_test_acc:
            best_test_acc = result['test_acc']
            epochs_no_improve = 0
            # Salvar o melhor modelo até agora
            torch.save(model.state_dict(), f'C:\\Users\\Olá\\programacao\\nasa\\New Plant Diseases Dataset(Augmented)\\model_epoch_{epoch+1}.pth')
            print(f"Modelo salvo: model_epoch_{epoch+1}.pth")
        else:
            epochs_no_improve += 1
            print(f"Acurácia no teste não melhorou por {epochs_no_improve} época(s).")
        
        # Verificar se devemos parar o treinamento
        if epochs_no_improve >= n_epochs_stop:
            print('Early stopping!')
            break
        
    return history

if __name__ == "__main__":

    start_time = time.time()
    
    print("Avaliando o modelo antes do treinamento...")
    history = [evaluate(model, valid_dataloader)]
    print("Acurácia e perda inicial na validação:", history)
    
    num_epoch = 50
    lr_rate = 0.01
    grad_clip = 0.15
    weight_decay = 1e-4
    optims = torch.optim.Adam

    print("Iniciando treinamento...")
    history += fit_OneCycle(num_epoch, lr_rate, model, train_dataloader, valid_dataloader, test_dataloader,
                            grad_clip=grad_clip, 
                            weight_decay=weight_decay, 
                            opt_func=optims)

    end_time = time.time()
    print(f"Treinamento concluído em {end_time - start_time:.2f} segundos")
