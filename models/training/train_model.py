"""
Script d'entraînement du modèle de reconnaissance de plats marocains
Utilise Transfer Learning avec MobileNetV2
"""

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import json
import os

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0001

# Chemin de la dataset (TES IMAGES SONT ICI)
DATASET_DIR = "../../dataset"

MODEL_SAVE_PATH = 'models/saved_models/moroccan_food_model.h5'


def create_model(num_classes):
    """
    Créer le modèle avec Transfer Learning
    """
    base_model = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )

    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model


def create_data_generators():
    """
    Créer les générateurs de données depuis dossier unique + split automatique
    """

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2   # <--- SPLIT AUTOMATIQUE TRAIN/VAL
    )

    # TRAIN
    train_generator = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    # VALIDATION
    val_generator = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    return train_generator, val_generator


def train_model():
    """
    Fonction principale d'entraînement
    """
    train_gen, val_gen = create_data_generators()

    num_classes = len(train_gen.class_indices)
    print(f"Nombre de classes: {num_classes}")
    print(f"Classes: {train_gen.class_indices}")

    # Sauvegarder le mapping
    with open('models/saved_models/class_indices.json', 'w', encoding='utf-8') as f:
        json.dump(train_gen.class_indices, f, ensure_ascii=False, indent=2)

    model = create_model(num_classes)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3)]
    )

    model.summary()

    callbacks = [
        ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]

    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )

    with open('models/saved_models/training_history.json', 'w') as f:
        json.dump(history.history, f, indent=2)

    print("Entraînement terminé !")
    print(f"Modèle sauvegardé dans : {MODEL_SAVE_PATH}")

    # ---- FINE-TUNING ----
    print("\n--- Début du Fine-Tuning ---")

    base_model = model.layers[0]
    base_model.trainable = True

    for layer in base_model.layers[:-20]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE / 10),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3)]
    )

    history_fine = model.fit(
        train_gen,
        epochs=20,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )

    print("Fine-tuning terminé !")


if __name__ == "__main__":
    os.makedirs('models/saved_models', exist_ok=True)
    train_model()
