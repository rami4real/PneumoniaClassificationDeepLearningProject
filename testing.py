from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
import numpy as np

# Load the saved model
model = load_model('pneumonia_detection_cnn_model.h5')

# Set the path to your test dataset
test_data_dir = './chest_xray/test'

# Set the image size based on your model's input size
img_size = (224, 224)

# Create an ImageDataGenerator for test data
test_datagen = ImageDataGenerator(rescale=1./255)

# Create a test generator
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=img_size,
    batch_size=32,  # Adjust based on your available memory
    class_mode='binary',  # Assuming binary classification (normal vs pneumonia)
    shuffle=False  # Do not shuffle the data
)

# Get true labels
y_true = test_generator.classes

# Predict using the model
y_pred_prob = model.predict(test_generator)
y_pred = np.argmax(y_pred_prob, axis=1)

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)

# Print the accuracy percentage
print(f'Accuracy: {accuracy * 100:.2f}%')
