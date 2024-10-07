import torch
import torchvision
import os
import numpy as np
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import box_iou

# Diretório de dados
root_dir = "C:/Users/Olá/programacao/nasa/leaf detectation"

# Dataset YOLO customizado
class LeafDataset(Dataset):
    def __init__(self, img_folder, transforms=None):
        self.img_folder = img_folder
        self.transforms = transforms
        self.imgs = [img for img in sorted(os.listdir(img_folder)) if img.endswith('.jpg')]
        self.labels = [os.path.splitext(img)[0] + '.txt' for img in self.imgs]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        label_name = self.labels[idx]
        img_path = os.path.join(self.img_folder, img_name)
        label_path = os.path.join(self.img_folder, label_name)
        
        # Verifica se o arquivo existe
        if not os.path.exists(img_path):
            print(f"Image file does not exist: {img_path}")
        if not os.path.exists(label_path):
            print(f"Label file does not exist: {label_path}")
        
        # Carrega a imagem e converte para RGB
        img = Image.open(img_path).convert('RGB')
        img_width, img_height = img.size  # Note: PIL usa (width, height)

        # Inicializa listas para armazenar as caixas e rótulos
        boxes = []
        labels = []
        
        # Lê o arquivo de rótulos
        with open(label_path, 'r') as file:
            for line in file.readlines():
                line = line.strip().split()
                class_id = int(line[0])
                x_center, y_center, width, height = map(float, line[1:])

                # Converte coordenadas do YOLO para formato de caixas delimitadoras
                x_min = (x_center - width / 2) * img_width
                y_min = (y_center - height / 2) * img_height
                x_max = (x_center + width / 2) * img_width
                y_max = (y_center + height / 2) * img_height

                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(1)  # Usando 1 para a classe "folha"

        # Converte para tensores
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels

        img = F.to_tensor(img)

        return img, target

# Função de colagem personalizada para o DataLoader
def collate_fn(batch):
    return tuple(zip(*batch))

# Função de treino
def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_sum = 0
    for batch_idx, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Calcula as perdas
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        loss_sum += losses.item()

        # Print a cada batch para acompanhar o progresso
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(data_loader)} | Loss: {losses.item():.4f}")

    avg_loss = loss_sum / len(data_loader)
    print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")
    return avg_loss

# Função de avaliação
@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    all_detections = []
    all_annotations = []
    for images, targets in data_loader:
        images = list(img.to(device) for img in images)
        outputs = model(images)
        for i in range(len(images)):
            preds = outputs[i]
            gt = targets[i]
            # Coletando detecções e anotações
            all_detections.append({'boxes': preds['boxes'].cpu(), 'scores': preds['scores'].cpu(), 'labels': preds['labels'].cpu()})
            all_annotations.append({'boxes': gt['boxes'], 'labels': gt['labels']})
    # Calcula mAP
    mAP = calculate_mAP(all_detections, all_annotations)
    return mAP

def calculate_mAP(detections, annotations, iou_threshold=0.5):
    # Cálculo simples de mAP para uma classe
    true_positives = []
    scores = []
    num_gt_boxes = 0

    for det, ann in zip(detections, annotations):
        gt_boxes = ann['boxes']
        num_gt_boxes += len(gt_boxes)
        detected = []
        pred_boxes = det['boxes']
        pred_scores = det['scores']

        for i in range(len(pred_boxes)):
            box = pred_boxes[i]
            score = pred_scores[i]
            scores.append(score)

            if len(gt_boxes) == 0:
                true_positives.append(0)
                continue

            ious = box_iou(box.unsqueeze(0), gt_boxes)
            max_iou, max_iou_idx = ious.max(1)
            if max_iou >= iou_threshold and max_iou_idx not in detected:
                true_positives.append(1)
                detected.append(max_iou_idx.item())
            else:
                true_positives.append(0)

    if len(true_positives) == 0:
        return 0.0

    # Ordena por score
    scores = torch.tensor(scores)
    true_positives = torch.tensor(true_positives)
    sorted_indices = torch.argsort(scores, descending=True)
    true_positives = true_positives[sorted_indices]

    # Calcula cumulativamente TP e FP
    cumulative_tp = torch.cumsum(true_positives, dim=0)
    cumulative_fp = torch.cumsum(1 - true_positives, dim=0)

    # Calcula precisão e recall
    precision = cumulative_tp / (cumulative_tp + cumulative_fp + 1e-6)
    recall = cumulative_tp / num_gt_boxes

    # Calcula AP
    ap = torch.trapz(precision, recall)

    return ap.item()

if __name__ == "__main__":
    # Datasets para treino, validação e teste
    train_dataset = LeafDataset(os.path.join(root_dir, "train"))
    valid_dataset = LeafDataset(os.path.join(root_dir, "valid"))
    test_dataset = LeafDataset(os.path.join(root_dir, "test"))

    # DataLoaders
    batch_size = 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

    # Modelo Faster R-CNN pré-treinado
    model = fasterrcnn_resnet50_fpn(weights='DEFAULT')

    # Ajusta a cabeça do modelo para o número de classes (1 classe de folha + fundo)
    num_classes = 2  # 1 classe (folha) + fundo
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    # Otimizador
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Envia o modelo para GPU, se disponível
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # Variável para armazenar o melhor mAP
    best_mAP = 0.0

    # Treinamento
    num_epochs = 10
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)

        print(f"Avaliando o modelo na validação após epoch {epoch+1}...")
        val_mAP = evaluate(model, valid_loader, device)
        print(f"Validation mAP: {val_mAP:.4f}")

        print(f"Avaliando o modelo no conjunto de teste após epoch {epoch+1}...")
        test_mAP = evaluate(model, test_loader, device)
        print(f"Test mAP: {test_mAP:.4f}")

        # Salvando o modelo após cada época
        model_save_path = f'C:\\Users\\Olá\\programacao\\nasa\\leaf detectation\\model_epoch_{epoch+1}.pth'
        torch.save(model.state_dict(), model_save_path)
        print(f"Modelo salvo em: {model_save_path}")

    print("Treinamento concluído.")
