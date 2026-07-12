import os
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# --- iskeletin başı csv okuma kısmı---
columns = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
data = pd.read_csv('data/driving_log.csv', names=columns)

print(f"Toplam örnek sayısı: {len(data)}")

# --- validation
train_samples, validation_samples = train_test_split(data, test_size=0.2, random_state=42)

print(f"Eğitim seti: {len(train_samples)} örnek")
print(f"Doğrulama seti: {len(validation_samples)} örnek")

# --- 3. Görüntü ön işleme fonksiyonu ---
def preprocess_image(img):
    img = img[60:135, :, :] #odak kısmı belirledim

    # NVIDIA modelinin makalesinde önerilen renk uzayı: YUV
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # NVIDIA beklediği sabit boy
    img = cv2.resize(img, (200, 66))

    return img

STEERING_CORRECTION = 0.2  # sol/sağ kamera için direksiyon düzeltme payı

def load_and_augment(sample):
    # sample: CSV'nin bir satırı (pandas Series)
    camera_choice = np.random.choice(['center', 'left', 'right'])
    img_path = sample[camera_choice].strip()
    steering = float(sample['steering'])

    # Sol/sağ kameraya göre direksiyon açısını düzelt
    if camera_choice == 'left':
        steering += STEERING_CORRECTION
    elif camera_choice == 'right':
        steering -= STEERING_CORRECTION

    img = cv2.imread(img_path)
    img = preprocess_image(img)

    # %50 ihtimalle görüntüyü yatayda aynala (flip) ve direksiyonu ters çevir
    if np.random.rand() < 0.5:
        img = cv2.flip(img, 1)
        steering = -steering

    return img, steering

def data_generator(samples, batch_size=32):
    samples = samples.reset_index(drop=True)
    num_samples = len(samples)

    while True:  # generator sonsuz döner, Keras kaç batch istediğini kendi ayarlar
        samples = shuffle(samples)

        for offset in range(0, num_samples, batch_size):
            batch_samples = samples.iloc[offset:offset + batch_size]

            images = []
            steerings = []

            for _, sample in batch_samples.iterrows():
                img, steering = load_and_augment(sample)
                images.append(img)
                steerings.append(steering)

            yield np.array(images), np.array(steerings)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, Lambda

def build_model():
    model = Sequential()

    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(66, 200, 3)))

    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))

    model.add(Flatten())


    model.add(Dropout(0.5))

    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))  # çıktı: tek sayı, direksiyon açısı (regresyon)

    model.compile(optimizer='adam', loss='mse')

    return model


model = build_model()
model.summary()

from tensorflow.keras.callbacks import ModelCheckpoint

# --- 6. Generator'ları oluştur ---
BATCH_SIZE = 32
train_generator = data_generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = data_generator(validation_samples, batch_size=BATCH_SIZE)

# --- Eğitim sırasında en iyi modeli otomatik kaydet ---
checkpoint = ModelCheckpoint(
    'model.h5',
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)

# --- 8. Eğitimi başlat ---
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_samples) // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=len(validation_samples) // BATCH_SIZE,
    epochs=10,
    callbacks=[checkpoint],
    verbose=1
)

print("\nEğitim tamamlandı! En iyi model 'model.h5' olarak kaydedildi.")