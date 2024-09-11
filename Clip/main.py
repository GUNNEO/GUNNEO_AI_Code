import __init__
import torch
import argparse
import models.build_model as build
import libs.text_preprocessing as text_preprocessing


'''default setting of the model'''
parser = argparse.ArgumentParser(description='Clip Network')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available else 'cpu'

'''hyperparameters setting'''
epoch = 0
max_epoch = 1000

modalities = ["T1", "T2"]


# teporally used for test model function
input_path = "/Users/gunneo/Downloads/乳腺癌北院_1.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image = torch.randn(1, 5, 3, 224, 224).to(device)
id, text = text_preprocessing.read_text_data(
    input_path, "影像号", "mri diagnosis report")
model = build.load_clip(num_img_modalities=2, pretrained=True)
text = text_preprocessing.preprocess_text(text[:5], model)
print(text.shape)
i, t = model(image, text)
print(i.shape, t.shape)
