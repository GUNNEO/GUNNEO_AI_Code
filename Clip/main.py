import __init__
import torch
import argparse
import libs.data_loader as data_loader
import libs.img_preprocessing as img_utils


'''default setting of the model'''
parser = argparse.ArgumentParser(description='Clip Network')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available else 'cpu'

'''hyperparameters setting'''
epoch = 0
max_epoch = 1000

modalities = ["T1", "T2"]


# teporally used for test model function
json_path = "/Users/gunneo/AI/Codes/Clip/datasets/hnscc.json"
output_path = "/Users/gunneo/AI/Codes/Clip/datasets/normalize_params.txt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_utils.cal_mean_std(json_file_path=json_path,
                       modality=1, output_path=output_path)
