import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
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


# Load training and test data
train_folder1 = './train1'
train_folder2 = './train2'
test_folder = './test'

X_data1, y_data1 = load_images_from_folder(train_folder1)
X_data2, y_data2 = load_images_from_folder(train_folder2)

# Normalize and reshape images
X_data1 = X_data1.astype('float32') / 255.
X_data2 = X_data2.astype('float32') / 255.
X_data1 = X_data1.reshape(-1, 28, 28, 1)
X_data2 = X_data2.reshape(-1, 28, 28, 1)
X_data = np.concatenate((X_data1, X_data2), axis=0)
y_data = np.concatenate((y_data1, y_data2), axis=0)

# One-hot encoding of the labels
label_binarizer = LabelBinarizer()
y_data = label_binarizer.fit_transform(y_data)

# Data augmentation setup
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    shear_range=0.1,
    fill_mode='nearest'
)


# Model definition with Batch Normalization
def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.5),
        Dense(len(label_binarizer.classes_), activation='softmax')  # Output layer for multi-class classification
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Cross-validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_no = 1
fold_accuracies = []

# Cross-validation training
for train_index, val_index in kf.split(X_data):
    print(f"Training on fold {fold_no}...")

    # Split the data
    X_train, X_val = X_data[train_index], X_data[val_index]
    y_train, y_val = y_data[train_index], y_data[val_index]

    # Data augmentation for training data
    train_generator = datagen.flow(X_train, y_train, batch_size=64)

    # Build the model
    model = build_model()

    # Early stopping and learning rate reduction
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)

    # Train the model
    history = model.fit(train_generator,
                        epochs=50,
                        validation_data=(X_val, y_val),
                        callbacks=[early_stopping, reduce_lr])

    # Evaluate the model on validation data
    val_loss, val_acc = model.evaluate(X_val, y_val)
    print(f"Validation accuracy for fold {fold_no}: {val_acc}")

    fold_accuracies.append(val_acc)
    fold_no += 1

# Print average accuracy
average_accuracy = np.mean(fold_accuracies)
print(f"Average cross-validation accuracy: {average_accuracy}")

# Model prediction on the test set
X_test, _ = load_images_from_folder(test_folder)
X_test = X_test.astype('float32') / 255.
X_test = X_test.reshape(-1, 28, 28, 1)

# Predict labels for the test set
y_test_pred = model.predict(X_test)

# Convert predictions back to original label format
y_test_labels = label_binarizer.inverse_transform(y_test_pred)

# Save predictions to a file
with open('predictions.txt', 'w') as f:
    for label in y_test_labels:
        f.write(f"{label}\n")
