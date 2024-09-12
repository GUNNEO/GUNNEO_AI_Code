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
image = torch.randn(1, 2, 3, 224, 224).to(device)
id, text = text_preprocessing.read_text_data(
    input_path, "影像号", "mri diagnosis report")
# text = ["i am good", "boy is a man", "yes, it is me",
#         "my name is gbionlp/bluebert_pubmed_mimic_uncased_L-24_H-1024_A-16unneo", "lego toys"]
model = build.load_clip(num_img_modalities=1,
                        text_pretrained=True, vision_pretrained=True)
text, mask = text_preprocessing.preprocess_text(
    texts=text[:2], model=model, pretrained="BlueBERT-L")
print(text.shape)
i, t = model(image, text, mask)
print(i.shape, t.shape)
