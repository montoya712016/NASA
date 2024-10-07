import os
from PIL import Image

def resize_images_in_folder(folder_path, output_size=(800, 800)):
    total_resized = 0
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(folder_path, filename)
            with Image.open(image_path) as img:
                img_resized = img.resize(output_size)
                img_resized.save(image_path)
                total_resized += 1
                print(f"Total de imagens redimensionadas: {total_resized}", end='\r')
    print(f"\nTotal de imagens redimensionadas em {folder_path}: {total_resized}")

if __name__ == "__main__":
    base_dir = "C:/Users/Olá/programacao/nasa/leaf detectation"
    for split in ['train', 'valid', 'test']:
        folder = os.path.join(base_dir, split)
        resize_images_in_folder(folder)
