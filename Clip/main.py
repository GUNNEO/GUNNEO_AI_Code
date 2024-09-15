import __init__
import torch
import argparse
import models.build_model as build
import libs.text_preprocessing as text_preprocessing
import libs.data_loader as data_loader


'''default setting of the model'''
parser = argparse.ArgumentParser(description='Clip Network')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available else 'cpu'

'''hyperparameters setting'''
epoch = 0
max_epoch = 1000

modalities = ["T1", "T2"]


# teporally used for test model function
text_path = "/Users/gunneo/Downloads/中山二院_单独MR_v5 2_1.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image = torch.randn(5, 1, 1, 224, 224).to(device)
data_dict = data_loader.read_data(
    text_path=text_path, **data_loader.return_hnscc_config())
print(data_dict[11089028]["img_path"][1])
