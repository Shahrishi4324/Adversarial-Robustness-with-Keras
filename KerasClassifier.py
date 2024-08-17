import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np

# Load and preprocess the dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = to_categorical(y_train), to_categorical(y_test)

# Build and compile the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Save the trained model
model.save('cifar10_cnn_model.h5')

# FGSM Attack
def fgsm_attack(image, epsilon, gradient):
    perturbation = epsilon * np.sign(gradient)
    adv_image = image + perturbation
    return np.clip(adv_image, 0, 1)

# Example FGSM attack
epsilon = 0.1
x_adv = fgsm_attack(x_test, epsilon, tf.gradients(model.output, model.input)[0])

# Evaluate the model on clean and adversarial examples
clean_accuracy = model.evaluate(x_test, y_test, verbose=0)[1]
adv_accuracy = model.evaluate(x_adv, y_test, verbose=0)[1]

print(f"Accuracy on clean examples: {clean_accuracy * 100:.2f}%")
print(f"Accuracy on adversarial examples: {adv_accuracy * 100:.2f}%")