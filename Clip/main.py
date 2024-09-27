import __init__
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import argparse
import platform
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Union, List
import libs.data_loader as data_loader
import models.build_model as build_model
import libs.text_preprocessing as text_utils
import models.image_encoder_backbone.attn_gan_fusion as agf


'''default setting of the model'''

home_path = Path(__file__).parent
system_name = platform.uname().system
json_path = "hnscc_sorted_1.json"


def parse_args():
    parser = argparse.ArgumentParser(description="config for model")
    # setting for data loader
    parser.add_argument('--json_file_path', type=str,
                        default=home_path / "datasets/" / json_path,
                        help="complete info to load all data")
    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--img_resolution', type=int, default=224)
    parser.add_argument('--img_num_slices', type=int, default=6)
    # setting for clip params
    parser.add_argument('--num_text_modalities', type=int, default=2)
    parser.add_argument('--num_img_modalities', type=int, default=2)
    parser.add_argument('--clip_params', type=dict,
                        default=build_model._clip_default_params())
    parser.add_argument('--text_pretrained', type=bool, default=False)
    parser.add_argument('--vision_pretrained', type=bool, default=False)
    parser.add_argument('--weights_conversion', type=int,
                        default=system_name in ["Linux", "Windows"])
    parser.add_argument('--img_fusion_method', type=str,
                        default="attn_gan_fusion")

    parser.add_argument('--epochs', type=int, default=1)
    # record the loss into log envry 10 batches
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--save_interval', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lambda_0', type=float, default=0.01)
    parser.add_argument('--gamma', type=int, default=10)
    parser.add_argument('--save_dir', type=str, default=home_path / "checkpoints/",
                        help="Directory to save the trained model.")
    parser.add_argument('--seed', type=int, default=405)
    parser.add_argument('--optimizer', type=str, default="AdamW")
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
    config.clip_params["img_fusion_method"] = config.img_fusion_method
    model = build_model.load_clip(
        num_text_modalities=config.num_text_modalities,
        num_img_modalities=config.num_img_modalities,
        text_pretrained=config.text_pretrained,
        vision_pretrained=config.vision_pretrained,
        clip_params=config.clip_params,
        device=device,
        weights_conversion=config.weights_conversion
    )

    # construct the criterion and optimizer
    optimizer_ls = []
    lr = config.lr
    if config.img_fusion_method == "multi_fusion":
        criterion = nn.CrossEntropyLoss().cuda(device)
        optimizer_name = config.optimizer
        if optimizer_name == "AdamW":
            optimizer = optim.AdamW(model.parameters(), lr=lr)
        elif optimizer_name == "SGD":
            optimizer = optim.SGD(
                model.parameters(), lr=lr, momentum=config.momentum)
        else:
            raise RuntimeError("invalid optimizer configuration")
        optimizer_ls.append(optimizer)
    elif config.img_fusion_method == "attn_gan_fusion":
        img_optimizer_ls = attn_gan_optimizer(
            config=config,
            fusion_model=model.vision_fusion,
            num_modalities=config.num_img_modalities,
            lr=lr
        )
        text_optimizer_ls = attn_gan_optimizer(
            config=config,
            fusion_model=model.text_fusion,
            num_modalities=config.num_text_modalities,
            lr=lr
        )
        clip_params = model.parameters()
        optimizer_name = config.optimizer
        optimizer_clip_ls = []
        if optimizer_name == "SGD":
            optimizer_clip_ls.append(
                optim.SGD(clip_params, lr=lr, momentum=config.momentum))
        elif optimizer_name == "AdamW":
            optimizer_clip_ls.append(optim.AdamW(clip_params, lr=lr))
        criterion = nn.CrossEntropyLoss().cuda(device)
        optimizer_ls = img_optimizer_ls + text_optimizer_ls + optimizer_clip_ls
    else:
        raise RuntimeError("invalid img fusion method")
    return dataloader, model, criterion, optimizer_ls, device


def attn_gan_optimizer(
    config: argparse.Namespace,
    fusion_model: nn.Module,
    num_modalities: int,
    lr: float
):
    if num_modalities == 1:
        return []
    params_D0 = []
    for m in range(num_modalities):
        params_D0 += list(fusion_model.D0[m].parameters())

    params_D1 = []
    for m in range(num_modalities):
        params_D1 += list(fusion_model.D1[m].parameters())

    params_Ds = []
    for m in range(num_modalities):
        params_Ds += list(fusion_model.Ds[m].parameters())

    params_Sm = [
        {'params': fusion_model.params_w, 'lr': lr},
        {'params': fusion_model.params_b, 'lr': lr},
        {'params': fusion_model.params_attn_w, 'lr': lr},
        {'params': fusion_model.params_attn_b, 'lr': lr},
    ]

    params_I = [
        {'params': fusion_model.param_w, 'lr': lr},
        {'params': fusion_model.param_b, 'lr': lr},
    ]

    fusion_params = params_D0 + params_D1 + params_Ds + params_I + params_Sm
    fusion_params = [p for sublist in fusion_params for p in sublist]

    optimizer_name = config.optimizer
    optimizer_ls = []
    if optimizer_name == "SGD":
        # optimizer for fusion part
        optimizer_D0 = optim.SGD(params_D0, lr=lr, momentum=config.momentum)
        optimizer_D1 = optim.SGD(params_D1, lr=lr, momentum=config.momentum)
        optimizer_Ds = optim.SGD(params_Ds, lr=lr, momentum=config.momentum)
        optimizer_Sm = optim.SGD(params_Sm, lr=lr, momentum=config.momentum)
        optimizer_I = optim.SGD(params_I, lr=lr, momentum=config.momentum)
        optimizer_ls.extend([optimizer_D0,
                            optimizer_D1, optimizer_Ds, optimizer_I, optimizer_Sm])
    elif optimizer_name == "AdamW":
        # optimizer for fusion part
        optimizer_D0 = optim.AdamW(params_D0, lr=lr)
        optimizer_D1 = optim.AdamW(params_D1, lr=lr)
        optimizer_Ds = optim.AdamW(params_Ds, lr=lr)
        optimizer_Sm = optim.AdamW(params_Sm, lr=lr)
        optimizer_I = optim.AdamW(params_I, lr=lr)
        optimizer_ls.extend([optimizer_D0,
                            optimizer_D1, optimizer_Ds, optimizer_I, optimizer_Sm])
    else:
        raise RuntimeError("invalid optimizer configuration")
    return optimizer_ls


def train(
    dataloader: data.DataLoader,
    model: nn.Module,
    criterion: nn.CrossEntropyLoss,
    optimizer_ls: List[Union[optim.SGD, optim.AdamW]],
    config: argparse.Namespace,
    device: torch.device
):
    model_save_dir = Path(config.save_dir) / config.model_name
    if not model_save_dir.exists():
        # Create a new folder if not exists
        model_save_dir.mkdir(exist_ok=True, parents=True)

    # Run training
    num_batches = len(dataloader)
    total_steps = num_batches * config.epochs
    example_ct = 0  # number of examples seen
    report_freq = config.log_interval
    for epoch in range(config.epochs):
        running_loss = 0.0  # running loss over batch
        for batch_idx, sample in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.epochs}")):
            current_step = epoch * num_batches + batch_idx
            p = current_step / total_steps
            lambda_adv = compute_lambda(
                p=p, lambda_0=config.lambda_0, gamma=config.gamma)
            # get the images
            images = sample["images"]
            print(images.shape)
            # (b, n_m, h, w, d) -> (n_m, b, d, h, w)

            texts = sample["texts"]
            texts, masks = text_utils.preprocess_text(texts=texts, model=model)

            # perform step for a single batch
            loss = train_batch(
                images=images,
                texts=texts,
                masks=masks,
                model=model,
                criterion=criterion,
                optimizer_ls=optimizer_ls,
                lambda_adv=lambda_adv,
                device=device
            )
            return
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


def compute_lambda(p, lambda_0, gamma=10):
    return lambda_0 * (2 / (1 + math.exp(-gamma * p)) - 1)


def auto_backward(
    optimizer_ls: List[Union[optim.AdamW, optim.SGD]],
    total_loss: float
):
    for optimizer in optimizer_ls:
        optimizer.zero_grad()
    total_loss.backward()

    for optimizer in optimizer_ls:
        optimizer.step()


def train_batch(
    images: torch.Tensor,
    texts: torch.Tensor,
    model: nn.Module,
    criterion: nn.CrossEntropyLoss,
    optimizer_ls: List[Union[optim.AdamW, optim.SGD]],
    lambda_adv: float,
    device: torch.device,
    masks: Optional[torch.Tensor] = None
):
    images, texts = images.to(device), texts.to(device)

    # Forward pass ->
    logits_per_image, logits_per_text, d0_img, d1_img, ds_img, d0_txt, d1_txt, ds_txt = model(
        images=images, texts=texts, masks=masks)

    # Create labels
    batch_size = images.shape[0]
    labels = torch.arange(batch_size).to(device)

    # Compute loss
    loss_img = criterion(logits_per_image, labels)
    loss_txt = criterion(logits_per_text, labels)
    if d0_img is None and d0_txt is None:
        total_loss = (loss_img + loss_txt) / 2  # avg. img and txt loss
    elif d0_img is not None and d0_txt is None:
        d0_loss, d1_loss, ds_loss = agf.compute_losses(
            d0=d0_img, d1=d1_img, ds=ds_img)
        total_loss = lambda_adv * \
            (d1_loss + ds_loss) / 2 + (1 - lambda_adv) * \
            (loss_txt + loss_img + d0_loss) / 3
    else:
        d0_img_loss, d1_img_loss, ds_img_loss = agf.compute_losses(
            d0=d0_img, d1=d1_img, ds=ds_img)
        d0_txt_loss, d1_txt_loss, ds_txt_loss = agf.compute_losses(
            d0=d0_txt, d1=d1_txt, ds=ds_txt)
        total_loss = lambda_adv * (d1_img_loss + d1_txt_loss + ds_img_loss + ds_txt_loss) / 4 + (
            1 - lambda_adv) * (loss_img + loss_txt + d0_img_loss + d0_txt_loss) / 4

    # Backward pass <-
    auto_backward(optimizer_ls=optimizer_ls, total_loss=total_loss)

    return total_loss


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
    dataloader, model, criterion, optimizer_ls, device = make(config)

    # train the model
    train(
        dataloader=dataloader,
        model=model,
        criterion=criterion,
        optimizer_ls=optimizer_ls,
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
