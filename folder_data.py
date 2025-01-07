import os
import shutil

train_dir = 'data/train'
test_dir = 'data/test1'
# Yeni alt klasörler oluştur
os.makedirs('data/train/cats', exist_ok=True)
os.makedirs('data/train/dogs', exist_ok=True)
os.makedirs('data/test1/cats', exist_ok=True)
os.makedirs('data/test1/dogs', exist_ok=True)

# Resimleri taşımak
for f in os.listdir(train_dir):
    if 'cat' in f.lower():
        shutil.move(os.path.join(train_dir, f), 'data/train/cats')
    elif 'dog' in f.lower():
        shutil.move(os.path.join(train_dir, f), 'data/train/dogs')

for f in os.listdir(test_dir):
    if 'cat' in f.lower():
        shutil.move(os.path.join(test_dir, f), 'data/test1/cats')
    elif 'dog' in f.lower():
        shutil.move(os.path.join(test_dir, f), 'data/test1/dogs')
        

import os
import shutil
from sklearn.model_selection import train_test_split

# Eğitim verisi klasörü
train_dir = "data/train"
val_dir = "data/val"

# Sınıfları kontrol et
classes = [cls for cls in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, cls))]

# Her sınıf için validation verisi oluştur
for cls in classes:
    class_path = os.path.join(train_dir, cls)
    images = os.listdir(class_path)
    train_images, val_images = train_test_split(images, test_size=0.2, random_state=42)

    # Validation klasörünü oluştur
    val_class_path = os.path.join(val_dir, cls)
    os.makedirs(val_class_path, exist_ok=True)

    # Resimleri taşı
    for img in val_images:
        shutil.move(os.path.join(class_path, img), os.path.join(val_class_path, img))

print("Validation verisi oluşturuldu ve taşındı!")