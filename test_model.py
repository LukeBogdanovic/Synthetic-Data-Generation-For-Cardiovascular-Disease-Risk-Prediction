import tensorflow as tf
from keras.api import layers, models
from keras.api.datasets import mnist
from keras.api.utils import to_categorical
from sklearn.metrics import roc_auc_score
import numpy as np


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
if len(tf.config.list_physical_devices('GPU')) < 1:
    exit()

# Load and preprocess data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize pixel values to be between 0 and 1
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# Reshape data to fit the model
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# One-hot encode labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Build model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

# Train model
model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.2)

# Evaluate model
test_loss, test_acc, test_auc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc:.2f}')
print(f'Test AUC: {test_auc:.2f}')

# Save model
model.save("mnist_cnn_model.h5")

# Calculate AUC using sklearn for more detailed analysis
predictions = model.predict(test_images)
# Convert labels to binary format for AUC calculation
test_labels_binary = np.argmax(test_labels, axis=1)
# Calculate AUC score using predicted probabilities
auc_score = roc_auc_score(test_labels_binary, predictions, multi_class='ovr')
print(f'Sklearn AUC: {auc_score:.2f}')