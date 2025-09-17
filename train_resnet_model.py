import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.optimizers import Adam

# --- Configuration ---
DATASET_PATH = 'dataset'
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
INITIAL_EPOCHS = 15 # Epochs for training just the top layer
FINE_TUNE_EPOCHS = 10 # Epochs for fine-tuning the whole model
NUM_CLASSES = 4 

def build_model():
    """
    Builds a classification model using ResNet50 as the base.
    """
    input_tensor = Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    
    base_model = ResNet50(
        weights='imagenet', 
        include_top=False, 
        input_tensor=input_tensor
    )
    
    print("Base model built successfully using ResNet50.")
    
    # Initially, we freeze the base model
    base_model.trainable = False

    # Add our custom layers on top
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model, base_model

def train():
    """
    Handles data loading, augmentation, and a two-stage training process.
    """
    # --- Data Augmentation ---
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )
    validation_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    # --- Prepare Data Iterators ---
    print("Loading training data...")
    train_generator = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )
    print("Loading validation data...")
    validation_generator = validation_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    model, base_model = build_model()

    # --- STAGE 1: Train only the top layers ---
    print("\n--- STAGE 1: Training Head ---")
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    history = model.fit(
        train_generator,
        epochs=INITIAL_EPOCHS,
        validation_data=validation_generator
    )

    # --- STAGE 2: Fine-tuning ---
    print("\n--- STAGE 2: Fine-tuning ---")
    # Unfreeze the base model to make it trainable
    base_model.trainable = True

    # Re-compile the model with a very low learning rate
    model.compile(optimizer=Adam(learning_rate=1e-5),  # 100x smaller learning rate
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    total_epochs = INITIAL_EPOCHS + FINE_TUNE_EPOCHS
    history_fine = model.fit(
        train_generator,
        epochs=total_epochs,
        initial_epoch=history.epoch[-1], # Continue from where we left off
        validation_data=validation_generator
    )

    print("\nModel training and fine-tuning completed.")

    # Save the final, fine-tuned model
    model.save('dermalscan_resnet_finetuned_model.h5')
    print("Model saved successfully as 'dermalscan_resnet_finetuned_model.h5'")

    # --- Plotting ---
    # Combine history objects for plotting
    acc = history.history['accuracy'] + history_fine.history['accuracy']
    val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
    loss = history.history['loss'] + history_fine.history['loss']
    val_loss = history.history['val_loss'] + history_fine.history['val_loss']

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.axvline(x=INITIAL_EPOCHS - 1, color='r', linestyle='--', label='Start Fine-Tuning')
    plt.title('Model Accuracy')
    plt.legend(loc='lower right')
    
    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.axvline(x=INITIAL_EPOCHS - 1, color='r', linestyle='--', label='Start Fine-Tuning')
    plt.title('Model Loss')
    plt.legend(loc='upper right')
    
    plt.show()

if __name__ == '__main__':
    train()

