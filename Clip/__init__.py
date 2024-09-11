import sys
import os


clip_path = os.path.dirname(os.path.abspath(__file__))
models_path = os.path.join(clip_path, "models")
libs_path = os.path.join(clip_path, "libs")
img_encoder_path = os.path.join(models_path, "image_encoder_backbone")
text_encoder_path = os.path.join(models_path, "text_encoder_backbone")

sys.path.append(clip_path)
sys.path.append(models_path)
sys.path.append(libs_path)
sys.path.append(img_encoder_path)
sys.path.append(text_encoder_path)
