import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os


# Assuming you have a PyTorch model class named MyModel
class MyModel(nn.Module):


# Define your model architecture here

# Load your PyTorch model
model = MyModel()
model.load_state_dict(torch.load("Torch.pth"))
model.eval()


# Define the preprocessing function for PyTorch
def preprocess_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    image = Image.open(img_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image


# Define the prediction function for PyTorch
def predict_pytorch(model, img):
    with torch.no_grad():
        preds = model(img)
    return preds.numpy()


# Define paths and files
txt_file1 = open("falseNormalPyTorch.txt", "w")
txt_file2 = open("falsePneumPyTorch.txt", "w")
img_path_normal = "./chest_xray/test/NORMAL"
img_path_pneumonia = "./chest_xray/test/PNEUMONIA"

# Evaluation for NORMAL images
correct_normal = 0
total_normal = 0

for img_name in sorted(os.listdir(img_path_normal)):
    if not img_name.lower().endswith(('.bmp', 'jpeg', 'jpg', 'png', 'tif', 'tiff')):
        continue
    filepath = os.path.join(img_path_normal, img_name)
    img = preprocess_image(filepath)
    total_normal += 1
    preds = predict_pytorch(model, img)

    if preds[0] >= 0.5:
        correct_normal += 1
    else:
        txt_file1.write(filepath + "\n")

# Evaluation for PNEUMONIA images
correct_pneumonia = 0
total_pneumonia = 0

for img_name in sorted(os.listdir(img_path_pneumonia)):
    if not img_name.lower().endswith(('.bmp', 'jpeg', 'jpg', 'png', 'tif', 'tiff')):
        continue
    filepath = os.path.join(img_path_pneumonia, img_name)
    img = preprocess_image(filepath)
    total_pneumonia += 1
    preds = predict_pytorch(model, img)

    if preds[1] >= 0.5:
        correct_pneumonia += 1
    else:
        txt_file2.write(filepath + "\n")

# Calculate accuracy
acc_normal = (correct_normal / total_normal) * 100
acc_pneumonia = (correct_pneumonia / total_pneumonia) * 100

# Print accuracies
print("Accuracy Normal: {:.2f}%".format(acc_normal))
print("Accuracy Pneumonia: {:.2f}%".format(acc_pneumonia))

# Close the text files
txt_file1.close()
txt_file2.close()
