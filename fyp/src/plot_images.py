import colorsys
import uuid
import os

import torch
import torch.nn.functional as F
import numpy as np

from pathlib import Path
from matplotlib import pyplot as plt


best_images = []

def plot_separate_images(
    epoch, sample, image, mavs, var, prediction_ss,
    prediction_ow, target, classes,
    plot_path='./plots', use_mav=True, delta=0.6
):
    var = {k: v.detach().clone() for k, v in var.items()}

    s_unk = contrastive_inference(prediction_ow)
    if use_mav and mavs is not None:
        s_sem, similarity = semantic_inference(prediction_ss, mavs, var)
        s_sem = s_sem.cuda()
        s_unk = (s_unk + s_sem) / 2

    ows_target = torch.zeros_like(target, dtype=torch.float64).to(target.device)
    ows_target[target == -1] = 1
    batch_errors = torch.mean(F.mse_loss((s_unk - 0.6).relu().bool().float(), ows_target, reduction='none'),dim=[1, 2]).detach().cpu().numpy()

    # ows_target = target.clone().cpu().numpy()
    ows_target *= 255
    ows_target = ows_target.cpu().numpy().astype(np.uint8)

    logits_ow = 255 * s_unk #(s_unk - 0.6).relu().bool().int()
    pred_ow = (s_unk - delta).relu().bool().int()
    pred_ow_np = pred_ow.cpu().numpy()
    logits_ow_np = logits_ow.cpu().numpy()

    deltas = [0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99]

    # Add images in the subsequent rows
    for idx in range(8):  # Limit to 8 rows of images
        if idx >= target.shape[0]:
            break
        current_id = str(uuid.uuid4())[:8]  # Generate a unique ID for each image
        img_dir = Path(plot_path) / "separate" / current_id

        if len(best_images) < 10:
            best_images.append([img_dir, float('inf')])
        else:
            errors = np.array([a[1] for a in best_images])
            if batch_errors[idx] < errors.max():
                min_idx = errors.argmax()
                best_images[min_idx] = [img_dir, batch_errors[idx]]
        
        img_dir.mkdir(parents=True, exist_ok=True)  # Create directory for each image
        if idx >= len(sample["image"]):  # Skip if less than 8 images in batch
            break

        # Normalize and process the input image
        image_np = image[idx].detach().cpu().permute(1, 2, 0).numpy()
        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())

        # Process prediction_ss (semantic segmentation prediction)
        pred_ss_np = (
            torch.argmax(torch.softmax(prediction_ss[idx], dim=0), dim=0)
            .detach()
            .cpu()
            .numpy()
        )

        # Process prediction_ow (open-world segmentation prediction)
        # pred_ow_np = prediction_ow[idx, 0].detach().cpu().numpy()
        target_copy = target.clone().cpu().numpy()

        # Process the ground truth
        target_np = target_copy[idx]

        target_c = np.zeros((*target_np.shape, 3), dtype=np.uint8)
        colors = generate_distinct_colors(classes)

        for i in range(classes):
            target_c[target_np == i] = colors[i]
        target_c = target_c.astype(np.uint8)

        pred_ss_np_c = np.zeros_like(target_c)
        for i in range(classes):
            pred_ss_np_c[pred_ss_np == i] = colors[i]
        pred_ss_np_c = pred_ss_np_c.astype(np.uint8)

        ows_binary_gt = ows_target[idx]

        # if use_mav:
        pred_ow_np_i = pred_ow_np[idx]
        logits_ow_np_i = logits_ow_np[idx]

        # Display Image, Prediction (SS), Prediction (OW), and Ground Truth
        write_image(image_np, img_dir, "display_image.png")

        write_image(pred_ss_np_c, img_dir, "prediction_ss.png")

        write_image(logits_ow_np_i, img_dir, "logits_ow.png")

        for de in deltas:
            pred_ow_np_i = (s_unk[idx] - de).relu().bool().int().cpu().numpy()
            write_image(pred_ow_np_i, img_dir, f"pred_ow_np_{de}.png")

        write_image(target_c, img_dir, "target_c.png")

        write_image(ows_binary_gt, img_dir, "ows_binary_gt.png")
    
    return best_images

def plot_images(
    epoch, sample, image, mavs, var, prediction_ss,
    prediction_ow, target, classes,
    plot_path='./plots', use_mav=True, delta=0.6
):
    # if plot_results and i < 1:  # Limit to first 8 samples for visualization
    # Create figure with 8 rows and 4 columns (adding a column for prediction_ow)
    fig, axes = plt.subplots(9, 6, figsize=(16, 24))  # One extra row for column names

    var = {k: v.detach().clone() for k, v in var.items()}

    # Add column names in the first row
    column_names = ["Image", "Prediction (SS)", "Logits (OW)", "Prediction (OW)", "Ground Truth", "OW Binary GT"]
    for col_idx, col_name in enumerate(column_names):
        axes[0, col_idx].text(
            0.5, 0.5, col_name, ha="center", va="center", fontsize=16, weight="bold"
        )
        axes[0, col_idx].axis("off")  # Turn off axes for header

    ows_target = target.clone().cpu().numpy()
    ows_target[ows_target == -1] = 255
    ows_target[ows_target < classes] = 0

    s_unk = contrastive_inference(prediction_ow)
    if use_mav and mavs is not None:
        s_sem, similarity = semantic_inference(prediction_ss, mavs, var)
        s_sem = s_sem.cuda()
        s_unk = (s_unk + s_sem) / 2
    logits_ow = 255 * s_unk #(s_unk - 0.6).relu().bool().int()
    pred_ow = (s_unk - delta).relu().bool().int()
    pred_ow_np = pred_ow.cpu().numpy()
    logits_ow_np = logits_ow.cpu().numpy()

    # Add images in the subsequent rows
    for idx in range(8):  # Limit to 8 rows of images
        if idx >= len(sample["image"]):  # Skip if less than 8 images in batch
            break

        # Normalize and process the input image
        image_np = image[idx].detach().cpu().permute(1, 2, 0).numpy()
        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())

        # Process prediction_ss (semantic segmentation prediction)
        pred_ss_np = (
            torch.argmax(torch.softmax(prediction_ss[idx], dim=0), dim=0)
            .detach()
            .cpu()
            .numpy()
        )

        # Process prediction_ow (open-world segmentation prediction)
        # pred_ow_np = prediction_ow[idx, 0].detach().cpu().numpy()
        target_copy = target.clone().cpu().numpy()

        # Process the ground truth
        target_np = target_copy[idx]

        target_c = np.zeros((*target_np.shape, 3), dtype=np.uint8)
        colors = generate_distinct_colors(classes)

        for i in range(classes):
            target_c[target_np == i] = colors[i]
        target_c = target_c.astype(np.uint8)

        pred_ss_np_c = np.zeros_like(target_c)
        for i in range(classes):
            pred_ss_np_c[pred_ss_np == i] = colors[i]
        pred_ss_np_c = pred_ss_np_c.astype(np.uint8)

        ows_binary_gt = ows_target[idx]

        # if use_mav:
        pred_ow_np_i = pred_ow_np[idx]
        logits_ow_np_i = logits_ow_np[idx]

        # Display Image, Prediction (SS), Prediction (OW), and Ground Truth
        axes[idx + 1, 0].imshow(image_np)
        axes[idx + 1, 0].axis("off")

        axes[idx + 1, 1].imshow(pred_ss_np_c)
        axes[idx + 1, 1].axis("off")

        # if use_mav:
        axes[idx + 1, 2].imshow(logits_ow_np_i)
        axes[idx + 1, 2].axis("off")

        # if use_mav:
        axes[idx + 1, 3].imshow(pred_ow_np_i)
        axes[idx + 1, 3].axis("off")

        axes[idx + 1, 4].imshow(target_c)
        axes[idx + 1, 4].axis("off")

        axes[idx + 1, 5].imshow(ows_binary_gt)
        axes[idx + 1, 5].axis("off")

    # Save the plot
    plot_dir = plot_path  # Ensure this is the directory path
    os.makedirs(plot_dir, exist_ok=True)  # Create the directory if it doesn't exist
    plot_file = f"val_imgs_epoch_{epoch}.png"
    full_plot_path = os.path.join(plot_dir, plot_file)  # Full path for the file

    # Save the plot
    plt.savefig(full_plot_path, bbox_inches="tight")
    plt.close(fig)

def write_image(image, dir, filename):
    # Display Image, Prediction (SS), Prediction (OW), and Ground Truth
    fig = plt.figure()
    plt.imshow(image)
    plt.axis("off")
    plt.savefig(dir / filename, bbox_inches="tight")
    plt.close(fig)

def generate_distinct_colors(n):
    colors = []
    for i in range(n):
        # Generate a color in HSV space and convert it to RGB
        hue = i / n  # equally spaced hue values
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)  # full saturation and value
        rgb = [int(c * 255) for c in rgb]  # Convert to 0-255 range
        colors.append(rgb)
    return colors


def contrastive_inference(predictions, radius=1.0):
    return F.relu(1 - torch.linalg.norm(predictions, dim=1) / radius)


def semantic_inference(predictions, mavs, var):
    mavs = mavs.to(predictions.device)
    stds = torch.vstack(tuple(var.values())).to(predictions.device)  # 19x19
    d_pred = (
        predictions[:, None, ...] - mavs[None, :, :, None, None]
    )  # [8,1,19,h,w] - [1,19,19,1,1]
    d_pred_ = d_pred / (stds[None, :, :, None, None] + 1e-8)
    scores = torch.exp(-torch.einsum("bcfhw,bcfhw->bchw", d_pred_, d_pred) / 2)
    best = scores.max(dim=1)
    return 1 - best[0], best[1]
