import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import os

# 1. Adatok betöltése és előfeldolgozása
X_data = np.load('train_images.npy')  # A képek adathalmaz
y_data = np.load('train_labels.npy')  # A címkék adathalmaz

# Normalizálás (0-255 -> 0-1 skálára hozva)
X_data = X_data.astype('float32') / 255.0

# Ha a képek mérete 28x28 és fekete-fehér
X_data = X_data.reshape(-1, 28, 28, 1)  # 1 csatornás képek (szürkeárnyalatos)

# Címkék kódolása (one-hot encoding)
lb = LabelBinarizer()
y_data = lb.fit_transform(y_data)  # Átalakítás one-hot címkékre

# Adatok felosztása tanuló és validációs halmazra
X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

# 2. Adat augmentáció alkalmazása
datagen = ImageDataGenerator(
    rotation_range=10,  # Forgatás +/- 10 fokkal
    width_shift_range=0.1,  # Vízszintes eltolás
    height_shift_range=0.1,  # Függőleges eltolás
    zoom_range=0.1  # Zoom
)
datagen.fit(X_train)

# 3. Modell felépítése
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.4),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),

    Dense(62, activation='softmax')  # 62 kimeneti kategória (0-9, a-z, A-Z)
])

# A modell fordítása
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 4. Modell tanítása
batch_size = 64
epochs = 50  # A epochs számot növelheted a jobb tanulás érdekében

history = model.fit(datagen.flow(X_train, y_train, batch_size=batch_size),
                    validation_data=(X_val, y_val),
                    steps_per_epoch=len(X_train) // batch_size,
                    epochs=epochs)

# 5. Modell értékelése és mentése
val_loss, val_acc = model.evaluate(X_val, y_val)
print(f"Validation Accuracy: {val_acc:.4f}, Validation Loss: {val_loss:.4f}")

# Modell mentése
model.save('character_recognition_model.h5')

# 6. Teszt adatok előrejelzése
X_test = np.load('test_images.npy')
X_test = X_test.astype('float32') / 255.0
X_test = X_test.reshape(-1, 28, 28, 1)

# Predikciók előállítása
y_pred = model.predict(X_test)

# Az előrejelzések legvalószínűbb kategóriájának kiválasztása
y_pred_classes = np.argmax(y_pred, axis=1)

# A predikciók visszaalakítása címkékké (például 0-9, a-z, A-Z)
y_pred_labels = lb.inverse_transform(y_pred_classes)

# 7. Teszt képek nevének betöltése
test_images_dir = 'path_to_test_images_directory'  # A teszt képek mappája
test_image_filenames = os.listdir(test_images_dir)

# 8. Predikciók mentése fájlba a kívánt formátumban
output_file = 'predictions.txt'
with open(output_file, 'w') as f:
    # Fejléc sor írása
    f.write("class;TestImage\n")

    for i, image_filename in enumerate(test_image_filenames):
        predicted_class = y_pred_classes[i]
        f.write(f"{predicted_class};{image_filename}\n")

print(f"Eredmények sikeresen kiírva a {output_file} fájlba.")
