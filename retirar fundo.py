from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from rembg import remove  # Biblioteca para remover fundo

# Carregar a imagem
image_path = "C:\\Users\\Olá\\programacao\\nasa\\New Plant Diseases Dataset(Augmented)\\train\\Tomato___Bacterial_spot\\0de30b71-8bd0-4270-a65b-7ae8befdd765___GCREC_Bact.Sp 6360.jpg"  # Coloque o caminho correto
image = Image.open(image_path)

# Remover o fundo da imagem usando rembg
# rembg.remove espera a imagem como bytes, então convertemos a imagem para bytes
with open(image_path, 'rb') as img_file:
    image_bytes = img_file.read()

# Aplicar a remoção do fundo
output = remove(image_bytes)

# Converter a saída em uma imagem PIL
image_no_bg = Image.open(BytesIO(output))

# Exibir a imagem original e a imagem sem fundo lado a lado
plt.figure(figsize=(10, 5))

# Imagem original
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Imagem Original")
plt.axis('off')

# Imagem sem fundo
plt.subplot(1, 2, 2)
plt.imshow(image_no_bg)
plt.title("Imagem Sem Fundo")
plt.axis('off')

plt.show()
