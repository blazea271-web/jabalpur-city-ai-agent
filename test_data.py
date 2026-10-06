import tensorflow as tf

# Load the dataset using Keras utility
dataset = tf.keras.utils.image_dataset_from_directory(
    "dataset",
    image_size=(224, 224),
    batch_size=4
)

# Print detected information
print("\n--- Dataset Summary ---")
print("Class names:", dataset.class_names)
print("Total number of batches:", len(dataset))

# Take 1 batch to verify image shape and labels
for images, labels in dataset.take(1):
    print(f"Batch image shape: {images.shape} (Batch, Height, Width, Channels)")
    print(f"Batch sample labels: {labels.numpy()}")