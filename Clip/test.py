from pathlib import Path as p

p1 = p("https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT/blob/main/pytorch_model.bin")
root = p.home() / ".cache" / "clip"
name = "BlueBert"+"_"+p1.name
print(root / name)
