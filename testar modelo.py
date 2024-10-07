import torch
import torchvision
from PIL import Image, ImageDraw
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F

# Função para carregar o modelo
def load_model(model_path, num_classes=2, device='cpu'):
    model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Função para realizar a predição e desenhar os retângulos
def predict_and_draw_boxes(model, image_path, device='cpu', threshold=0.5):
    # Carregar a imagem
    img = Image.open(image_path).convert('RGB')
    img_tensor = F.to_tensor(img).unsqueeze(0).to(device)

    # Fazer a predição
    with torch.no_grad():
        predictions = model(img_tensor)

    # Obter as caixas e as pontuações
    pred_boxes = predictions[0]['boxes'].cpu().numpy()
    pred_scores = predictions[0]['scores'].cpu().numpy()

    print(predictions)

    # Desenhar as caixas na imagem
    draw = ImageDraw.Draw(img)
    for i, box in enumerate(pred_boxes):
        if pred_scores[i] >= threshold:  # Aplicar o threshold
            draw.rectangle(box, outline='red', width=3)

    return img

if __name__ == "__main__":
    # Definir o caminho do modelo treinado e da imagem para teste
    model_path = 'C:/Users/Olá/programacao/nasa/leaf detectation/model_epoch_5.pth'
    test_image_path = 'C:/Users/Olá/Downloads/5nTXLbgT69qWcndC7vk6TH.jpg'

    # Definir o dispositivo (CPU ou GPU)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Carregar o modelo treinado
    model = load_model(model_path, num_classes=2, device=device)

    # Realizar a predição e desenhar os retângulos
    result_image = predict_and_draw_boxes(model, test_image_path, device=device, threshold=0.5)

    # Exibir a imagem resultante com as caixas
    result_image.show()  # Exibe a imagem
