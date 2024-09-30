import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import KFold
from tensorflow.keras.regularizers import l2
import os
from PIL import Image

def load_images_from_folder(folder):
    images = []
    labels = []
    for subdir in os.listdir(folder):
        subdir_path = os.path.join(folder, subdir)
        if os.path.isdir(subdir_path):
            for filename in os.listdir(subdir_path):
                img_path = os.path.join(subdir_path, filename)
                img = Image.open(img_path).convert('L')  # Convert to grayscale
                img = img.resize((28, 28))  # Resize to 28x28 pixels
                img_array = np.array(img)
                images.append(img_array)
                labels.append(subdir)  # Folder name is the label
    return np.array(images), np.array(labels)

train_folder1 = './train1'
train_folder2 = './train2'

X_data1, y_data1 = load_images_from_folder(train_folder1)
X_data2, y_data2 = load_images_from_folder(train_folder2)

# Combine data
X_data = np.concatenate((X_data1, X_data2), axis=0)
y_data = np.concatenate((y_data1, y_data2), axis=0)

# Normalize (0-255 -> 0-1 scale)
X_data = X_data.astype('float32') / 255.0

# Reshape if images are 28x28 and grayscale
X_data = X_data.reshape(-1, 28, 28, 1)  # 1 channel images (grayscale)

# Encode labels (one-hot encoding)
lb = LabelBinarizer()
y_data = lb.fit_transform(y_data)  # Convert to one-hot labels

# Set up cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Build model
def create_model():
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1), kernel_regularizer=l2(0.001)),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        Conv2D(256, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
        MaxPooling2D((2, 2)),
        Dropout(0.4),

        Flatten(),
        Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.5),

        Dense(62, activation='softmax')  # 62 output classes (0-9, a-z, A-Z)
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Cross-validation and model training
fold_no = 1
for train_index, val_index in kf.split(X_data):
    X_train, X_val = X_data[train_index], X_data[val_index]
    y_train, y_val = y_data[train_index], y_data[val_index]

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=10,  # Rotate +/- 10 degrees
        width_shift_range=0.1,  # Horizontal shift
        height_shift_range=0.1,  # Vertical shift
        zoom_range=0.1,  # Zoom
        horizontal_flip=True  # Horizontal flip
    )
    datagen.fit(X_train)

    model = create_model()

    # Callbacks for early stopping and learning rate reduction
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

    # Train model
    batch_size = 64
    epochs = 100  # Increase epochs for better learning
    train_data = datagen.flow(X_train, y_train, batch_size=batch_size)
    history = model.fit(train_data,
                        validation_data=(X_val, y_val),
                        steps_per_epoch=len(X_train) // batch_size,
                        epochs=epochs,
                        callbacks=[early_stopping, reduce_lr])

    # Evaluate model
    val_loss, val_acc = model.evaluate(X_val, y_val)
    print(f"Fold {fold_no} - Validation Accuracy: {val_acc:.4f}, Validation Loss: {val_loss:.4f}")
    fold_no += 1

# Save model
model.save('character_recognition_model.keras')

# Load test data
test_folder = './test'
X_test = []
test_image_filenames = []

for filename in os.listdir(test_folder):
    img_path = os.path.join(test_folder, filename)
    img = Image.open(img_path).convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to 28x28 pixels
    img_array = np.array(img)
    X_test.append(img_array)
    test_image_filenames.append(filename)

X_test = np.array(X_test)
X_test = X_test.astype('float32') / 255.0
X_test = X_test.reshape(-1, 28, 28, 1)

# Generate predictions
y_pred = model.predict(X_test)

# Ensure y_pred is a 2D array
if len(y_pred.shape) == 1:
    y_pred = y_pred.reshape(-1, 1)

# Select the most likely category for each prediction
y_pred_classes = np.argmax(y_pred, axis=1)

# Convert predictions back to labels (e.g., 0-9, a-z, A-Z)
y_pred_labels = lb.inverse_transform(y_pred_classes)

# Save predictions to file in the desired format
output_file = 'predictions.txt'
with open(output_file, 'w') as f:
    # Write header row
    f.write("class;TestImage\n")

    # Write predictions and filenames
    for i, image_filename in enumerate(test_image_filenames):
        predicted_class = y_pred_classes[i]
        f.write(f"{predicted_class};{image_filename}\n")

print(f"Results successfully written to {output_file}.")