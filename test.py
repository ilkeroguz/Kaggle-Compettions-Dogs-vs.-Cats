import os
import shutil
import torch
from PIL import Image
from utils import CNNCatsDogs, get_transform

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
model_path = 'cat_dog_classifier.pth'
test_dir = 'data/test1'  # Test verilerinin bulunduğu klasör
output_dirs = {'dogs': 'data/test1/dogs', 'cats': 'data/test1/cats'}

# Klasörlerin oluşturulması
for output_dir in output_dirs.values():
    os.makedirs(output_dir, exist_ok=True)

# Görüntü işleme
transform = get_transform()

# Model yükleme
model = CNNCatsDogs().to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# Test verilerinin işlenmesi
for image_name in os.listdir(test_dir):
    image_path = os.path.join(test_dir, image_name)

    # Sadece dosya olan girişlerle ilgilen
    if not os.path.isfile(image_path):
        continue

    # Görüntüyü yükleme ve ön işleme
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Modelin tahmini
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        label = 'dogs' if predicted.item() == 0 else 'cats'

    # Görüntüyü ilgili klasöre taşıma
    shutil.move(image_path, os.path.join(output_dirs[label], image_name))

print("Tüm test görüntüleri başarıyla sınıflandırıldı ve taşındı.")
