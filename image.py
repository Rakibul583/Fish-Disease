import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Set dataset directory (raw string literal to handle Windows paths)
dataset_dir = r'C:\Users\Student\Desktop\Rakibul\Surayea'

# Check if the directory exists
if not os.path.exists(dataset_dir):
    print(f"Error: The directory {dataset_dir} does not exist.")
else:
    print(f"Directory {dataset_dir} found!")

# Set up ImageDataGenerators for training and validation data
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize the image pixels to [0, 1]
    rotation_range=40,  # Random rotations
    width_shift_range=0.2,  # Random horizontal shift
    height_shift_range=0.2,  # Random vertical shift
    shear_range=0.2,  # Random shear
    zoom_range=0.2,  # Random zoom
    horizontal_flip=True,  # Random horizontal flip
    fill_mode='nearest'  # Fill mode for missing pixels during transformations
)

val_datagen = ImageDataGenerator(rescale=1./255)  # Only rescale for validation

# Define the training data generator
train_generator = train_datagen.flow_from_directory(
    dataset_dir,  # The parent folder containing Argulus and Broken antennae and rostrum
    target_size=(224, 224),  # Resize images to 224x224
    batch_size=32,
    class_mode='categorical'  # Use categorical cross-entropy for multi-class classification
)

# Define the validation data generator
validation_generator = val_datagen.flow_from_directory(
    dataset_dir,  # The parent folder containing Argulus and Broken antennae and rostrum
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'  # Categorical classification
)

# Define the model using Transfer Learning (VGG16)
base_model = tf.keras.applications.VGG16(
    weights='imagenet',  # Use pre-trained weights from ImageNet
    include_top=False,  # Exclude the top layer (we'll add our own)
    input_shape=(224, 224, 3)  # Image input shape (224x224, 3 channels)
)

# Freeze the base model to avoid retraining it
base_model.trainable = False

# Create a new model by adding custom layers on top of VGG16
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),  # Reduce dimensions for the fully connected layers
    layers.Dense(256, activation='relu'),  # Fully connected layer with 256 units
    layers.Dense(train_generator.num_classes, activation='softmax')  # Output layer for classification
])

# Compile the model
model.compile(
    optimizer='adam',  # Adam optimizer
    loss='categorical_crossentropy',  # Loss function for multi-class classification
    metrics=['accuracy']  # Accuracy metric
)

# Train the model
history = model.fit(
    train_generator,  # Training data
    steps_per_epoch=train_generator.samples // train_generator.batch_size,  # Number of steps per epoch
    epochs=10,  # Number of epochs
    validation_data=validation_generator,  # Validation data
    validation_steps=validation_generator.samples // validation_generator.batch_size  # Number of validation steps
)

# Save the model for future use
model.save('argulus_and_broken_classifier.h5')  # Save the model to a file

# Optional - Visualize the Training Process
plt.figure(figsize=(12, 6))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
