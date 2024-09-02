import tensorflow as tf
import numpy as np
import os
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Resizing, Rescaling
from tensorflow.keras.regularizers import l2

# Constants for image dimensions and training
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
BATCH_SIZE = 32
EPOCHS = 20

# Paths for training and testing datasets
BASE_DIR = '/Users/balajisivakumar/Downloads/'
TRAIN_DIR = os.path.join(BASE_DIR, 'Images', 'train')
TEST_DIR = os.path.join(BASE_DIR, 'Images', 'test')

# Load the training, validation, and test datasets
train_data = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    color_mode='rgb',
    image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    shuffle=True,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="training",
    seed=42
)

validation_data = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    color_mode='rgb',
    image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    shuffle=True,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="validation",
    seed=42
)

test_data = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    color_mode='rgb',
    image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    shuffle=True,
    batch_size=BATCH_SIZE,
    seed=42
)

# Get class names from the training dataset
class_labels = train_data.class_names

# Define a function to build the CNN model
def build_cnn_model(input_dim, num_classes):
    """
    Constructs and compiles a convolutional neural network model.

    Parameters:
        input_dim (tuple): Dimensions of the input images.
        num_classes (int): Number of output classes.

    Returns:
        model (tf.keras.Model): Compiled CNN model.
    """
    model = Sequential([
        Resizing(IMAGE_HEIGHT, IMAGE_WIDTH, input_shape=input_dim),
        Rescaling(1.0 / 255),

        #First Convolution Block
        Conv2D(32, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),   
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),

        #Second Convolution Block
        Conv2D(32, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),

        #Final Convolution Block
        Conv2D(32, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        
        Dense(1500, activation='relu'),
        Dense(100, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# Initialize the enhanced model
input_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, 3)
cnn_model = build_cnn_model(input_shape, len(class_labels))


# Train the model on the training dataset and validate on the validation dataset
def train_model(model, train_dataset, val_dataset, epochs):
    """
    Trains the model on the provided training and validation datasets.
    
    Parameters:
        model (tf.keras.Model): The CNN model to train.
        train_dataset (tf.data.Dataset): The training dataset.
        val_dataset (tf.data.Dataset): The validation dataset.
        epochs (int): Number of epochs for training.
    
    Returns:
        history (tf.keras.callbacks.History): Training history object.
    """
    history = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs)
    return history

# Train the model without explicitly specifying a device
training_history = train_model(cnn_model, train_data, validation_data, EPOCHS)

# Evaluate the model on the test dataset
def evaluate_model(model, test_dataset):
    """
    Evaluates the model on the test dataset.
    
    Parameters:
        model (tf.keras.Model): The CNN model to evaluate.
        test_dataset (tf.data.Dataset): The test dataset.
    
    Returns:
        test_accuracy (float): Accuracy of the model on the test dataset.
    """
    test_loss, test_accuracy = model.evaluate(test_dataset, verbose=2)
    print(f'\nTest accuracy: {test_accuracy:.4f}')
    return test_accuracy

test_accuracy = evaluate_model(cnn_model, test_data)

# Plot the training and validation accuracy and loss
def plot_training_results(history, epochs):
    """
    Plots the training and validation accuracy and loss over epochs.
    
    Parameters:
        history (tf.keras.callbacks.History): Training history object.
        epochs (int): Number of epochs for training.
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epoch_range = range(epochs)

    plt.figure(figsize=(12, 5))

    # Plotting Training Accuracy
    plt.subplot(1, 2, 1)
    plt.bar(epoch_range, acc, label='Training Accuracy', color='red', alpha=0.6, width=0.4, align='center')
    plt.bar(epoch_range, val_acc, label='Validation Accuracy', color='black', alpha=0.6, width=0.4, align='edge')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    # Plotting Training and Validation Loss using bar charts
    plt.subplot(1, 2, 2)
    plt.bar(epoch_range, loss, label='Training Loss', color='orange', alpha=0.6, width=0.4, align='center')
    plt.bar(epoch_range, val_loss, label='Validation Loss', color='green', alpha=0.6, width=0.4, align='edge')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    plt.tight_layout()
    plt.show()
    
plot_training_results(training_history, EPOCHS)

# Print the model summary
cnn_model.summary()

save_directory = os.path.join(BASE_DIR, 'Images', 'kitchenware_trainedmodel.h5')
# Save the trained model
def save_trained_model(model, save_directory):
    """
    Saves the trained model to the specified directory.
    
    Parameters:
        model (tf.keras.Model): The trained model to save.
        save_directory (str): Directory path where the model will be saved.
    """
    os.makedirs(save_directory, exist_ok=True)
    #Saving the model in the native Keras format
    model.save('/Users/balajisivakumar/Downloads/TrainedModel/kitchenware_trainedmodel.keras')
    print(f'Model saved to {os.path.join(save_directory, "kitchenware_trainedmodel.keras")}')

save_path = os.path.join(BASE_DIR, 'TrainedModel')
save_trained_model(cnn_model, save_path)