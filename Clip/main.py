import __init__
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import argparse
import platform
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Union
import libs.data_loader as data_loader
import models.build_model as build_model
import libs.text_preprocessing as text_utils


'''default setting of the model'''

home_path = Path(__file__).parent
system_name = platform.uname().system


def parse_args():
    parser = argparse.ArgumentParser(description="config for model")
    # setting for data loader
    parser.add_argument('--json_file_path', type=str,
                        default=home_path / "datasets/hnscc_sorted.json",
                        help="complete info to load all data")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--img_resolution', type=int, default=224)
    parser.add_argument('--img_num_slices', type=int, default=1)
    # setting for clip params
    parser.add_argument('--num_img_modalities', type=int, default=2)
    parser.add_argument('--clip_params', type=dict,
                        default=build_model._clip_default_params())
    parser.add_argument('--text_pretrained', type=bool, default=False)
    parser.add_argument('--vision_pretrained', type=bool, default=False)
    parser.add_argument('--weights_conversion', type=int,
                        default=system_name in ["Linux", "Windows"])

    parser.add_argument('--epochs', type=int, default=200)
    # record the loss into log envry 10 batches
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--save_interval', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_dir', type=str, default=home_path / "checkpoints/",
                        help="Directory to save the trained model.")
    parser.add_argument('--seed', type=int, default=405)
    parser.add_argument('--optimizer', type=str, default="SGD")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--model_name', type=str, default="PC_clip")
    args = parser.parse_args()
    return args


def make(
    config: argparse.Namespace
):
    # load data here
    dataloader, device = data_loader.load_data(
        json_file_path=config.json_file_path,
        batch_size=config.batch_size,
        img_num_slices=config.img_num_slices,
        img_resolution=config.img_resolution
    )

    # make the model here
    config.clip_params["vision_channels"] = config.img_num_slices
    model = build_model.load_clip(
        num_img_modalities=config.num_img_modalities,
        text_pretrained=config.text_pretrained,
        vision_pretrained=config.vision_pretrained,
        clip_params=config.clip_params,
        device=device,
        weights_conversion=config.weights_conversion
    )

    # construct the criterion and optimizer
    criterion = nn.CrossEntropyLoss().cuda(device)
    optimizer_name = config.optimizer
    if optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(
            model.parameters(), lr=config.lr, momentum=config.momentum)
    else:
        raise RuntimeError("invalid optimizer configuration")
    return dataloader, model, criterion, optimizer, device


def train(
    dataloader: data.DataLoader,
    model: nn.Module,
    criterion: nn.CrossEntropyLoss,
    optimizer: optim,
    config: argparse.Namespace,
    device: torch.device
):
    model_save_dir = Path(config.save_dir) / config.model_name
    if not model_save_dir.exists():
        # Create a new folder if not exists
        model_save_dir.mkdir(exist_ok=True, parents=True)

    # Run training
    num_batches = len(dataloader)
    example_ct = 0  # number of examples seen
    report_freq = config.log_interval
    for epoch in range(config.epochs):
        running_loss = 0.0  # running loss over batch
        for batch_idx, sample in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.epochs}")):
            # get the images
            images = sample["images"]
            # (b, n_m, h, w, d) -> (n_m, b, d, h, w)

            texts = sample["texts"]
            texts, masks = text_utils.preprocess_text(texts=texts, model=model)

            # temporally change the size of text
            texts = texts.squeeze()
            if masks is not None:
                masks = masks.squeeze()

            # perform step for a single batch
            loss = train_batch(
                images=images,
                texts=texts,
                masks=masks,
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                device=device
            )
            example_ct += images.shape[1]
            running_loss += loss.item()

            # Report metrics every `report_freq` batch
            if ((batch_idx + 1) % report_freq) == 0 or batch_idx == num_batches - 1:
                train_log(running_loss / report_freq, example_ct, epoch)
                running_loss = 0.0

            if ((epoch + 1) % config.save_interval) == 0:
                model_path = Path(model_save_dir) / \
                    f"checkpoint_{epoch+1}.pt"
                print("Saved checkpoint to: ", model_path)
                save(model, model_path)


def train_batch(
    images: torch.Tensor,
    texts: torch.Tensor,
    model: nn.Module,
    criterion: nn.CrossEntropyLoss,
    optimizer: Union[optim.AdamW, optim.SGD],
    device: torch.device,
    masks: Optional[torch.Tensor] = None
):
    images, texts = images.to(device), texts.to(device)

    # Forward pass ->
    logits_per_image, logits_per_text = model(
        images=images, texts=texts, masks=masks)

    # Create labels
    batch_size = images.shape[0]
    labels = torch.arange(batch_size).to(device)

    # Compute loss
    loss_img = criterion(logits_per_image, labels)
    loss_txt = criterion(logits_per_text, labels)
    loss = (loss_img + loss_txt) / 2  # avg. img and txt loss

    # Backward pass <-
    optimizer.zero_grad()
    loss.backward()

    # Step with optimizer
    optimizer.step()

    return loss


def train_log(
    loss: torch.float16,
    example_ct: int,
    epoch: int
):
    loss = float(loss)
    print(f"Loss after {str(example_ct).zfill(5)}" + f" examples: {loss:.3f}")


def save(
    model: nn.Module,
    path: str
):
    torch.save(model.state_dict(), path)


def model_pipeline(
    config: argparse.Namespace,
    verbose: bool = False
):
    # make the model, data, and optimization problem
    dataloader, model, criterion, optimizer, device = make(config)

    # train the model
    train(
        dataloader=dataloader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        config=config,
        device=device
    )

    # save model
    model_path = Path(config.save_dir) / f"{config.model_name}_checkpoint.pt"
    save(model, model_path)

    if verbose:
        print(model)
    return model


if __name__ == "__main__":
    args = parse_args()
    model = model_pipeline(config=args)
