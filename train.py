import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Load dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset",
    image_size=(128, 128),
    batch_size=4
)

class_names = train_ds.class_names
print("\nLandmarks to learn:", class_names)

# 2. Upgraded CNN with Data Augmentation
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(128, 128, 3)),

    # 🔄 Data Augmentation: Learns from mirrored photos too!
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),

    # CNN Layers
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'), # More filters for details
    layers.MaxPooling2D((2, 2)),

    # Decision Head
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),                            # Prevents memorizing mistakes
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 🚀 epochs means how many time the model will train from the data
print("\n🚀 Training with 50 Epochs...")
model.fit(train_ds, epochs=50)

# Save
model.save("jabalpur_cnn.keras")
print("\n🎉 High-Accuracy CNN saved to 'jabalpur_cnn.keras'!")
