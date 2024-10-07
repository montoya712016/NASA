import os
import random
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from io import BytesIO
import matplotlib.pyplot as plt
from rembg import remove

# Caminhos
train_dir = "C:/Users/Olá/programacao/nasa/New Plant Diseases Dataset(Augmented)/train"
fundos_path = "C:/Users/Olá/programacao/nasa/New Plant Diseases Dataset(Augmented)/unsplash-images-collection"

# Função para escolher uma folha aleatória de qualquer classe dentro do diretório de treinamento
def escolher_folha_aleatoria(train_dir):
    pasta_classe = random.choice(os.listdir(train_dir))
    classe_path = os.path.join(train_dir, pasta_classe)
    folha_image_path = os.path.join(classe_path, random.choice(os.listdir(classe_path)))
    return folha_image_path

# Função para aplicar transformações (brilho, contraste, ruído, etc)
def aplicar_transformacoes(imagem):
    angulo_rotacao = random.randint(0, 360)
    imagem = imagem.rotate(angulo_rotacao, expand=True)
    
    enhancer_brilho = ImageEnhance.Brightness(imagem)
    fator_brilho = random.uniform(0.8, 1.2) 
    imagem = enhancer_brilho.enhance(fator_brilho)
    
    enhancer_contraste = ImageEnhance.Contrast(imagem)
    fator_contraste = random.uniform(0.8, 1.3) 
    imagem = enhancer_contraste.enhance(fator_contraste)
    
    enhancer_cor = ImageEnhance.Color(imagem)
    fator_cor = random.uniform(0.8, 1.2) 
    imagem = enhancer_cor.enhance(fator_cor)
    
    imagem = imagem.filter(ImageFilter.GaussianBlur(random.uniform(0, 0.5)))  
    return imagem

# Função para verificar sobreposição
def verificar_sobreposicao(nova_caixa, caixas_existentes):
    for caixa in caixas_existentes:
        # Verifica se as caixas se sobrepõem
        if not (nova_caixa[2] < caixa[0] or nova_caixa[0] > caixa[2] or nova_caixa[3] < caixa[1] or nova_caixa[1] > caixa[3]):
            return True
    return False

# Função principal
def criar_montagem_com_folhas(fundos_path, train_dir, num_folhas=1):
    # Escolher fundo aleatório
    fundo_aleatorio_path = os.path.join(fundos_path, random.choice(os.listdir(fundos_path)))
    fundo_image = Image.open(fundo_aleatorio_path)
    fundo_image = fundo_image.resize((1024, 1024))
    fundo_image = aplicar_transformacoes(fundo_image)
    
    caixas_delimitadoras = []
    draw = ImageDraw.Draw(fundo_image)
    
    for _ in range(num_folhas):
        # Carregar uma folha aleatória
        folha_image_path = escolher_folha_aleatoria(train_dir)
        with open(folha_image_path, 'rb') as img_file:
            folha_image_bytes = img_file.read()
        folha_image_sem_fundo = remove(folha_image_bytes)
        folha_image_sem_fundo = Image.open(BytesIO(folha_image_sem_fundo))
        
        # Aplicar transformações na folha
        folha_image_sem_fundo = aplicar_transformacoes(folha_image_sem_fundo)
        
        # Redimensionar a folha
        folha_proporcao = folha_image_sem_fundo.size[0] / folha_image_sem_fundo.size[1]
        nova_largura = random.randint(256, 256)
        nova_altura = int(nova_largura / folha_proporcao)
        folha_image_sem_fundo = folha_image_sem_fundo.resize((nova_largura, nova_altura))
        
        # Tentar encontrar uma posição sem sobreposição
        for tentativa in range(10):
            pos_x = random.randint(0, fundo_image.size[0] - folha_image_sem_fundo.size[0])
            pos_y = random.randint(0, fundo_image.size[1] - folha_image_sem_fundo.size[1])
            
            nova_caixa = [pos_x, pos_y, pos_x + folha_image_sem_fundo.size[0], pos_y + folha_image_sem_fundo.size[1]]
            
            if not verificar_sobreposicao(nova_caixa, caixas_delimitadoras):
                caixas_delimitadoras.append(nova_caixa)
                fundo_image.paste(folha_image_sem_fundo, (pos_x, pos_y), folha_image_sem_fundo)
                draw.rectangle(nova_caixa, outline="red", width=3)
                break
    
    return fundo_image

# Gerar uma montagem com 1 a 4 folhas
num_folhas = random.randint(1, 4)
montagem_final = criar_montagem_com_folhas(fundos_path, train_dir, num_folhas)

# Exibir a imagem final
plt.figure(figsize=(8, 8))
plt.imshow(montagem_final)
plt.title(f"Montagem com {num_folhas} Folhas e Caixa Delimitadora")
plt.axis('off')
plt.show()
