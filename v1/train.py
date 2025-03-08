# Copyright (C) 2024. All rights reserved.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from datetime import datetime
import json
import os
import copy
import sys
import time
import warnings
import matplotlib.pyplot as plt
import pickle

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.utils import make_grid
import torch.optim
from torch.optim.lr_scheduler import OneCycleLR
from src.args import ArgumentParser
from src.build_model import build_model
from src import utils
from src import losses
from src.prepare_data import prepare_data
from src.utils import save_ckpt_every_epoch
from src.utils import load_ckpt
from src.utils import print_log

from sklearn.cluster import AgglomerativeClustering as ac
from torchmetrics import JaccardIndex as IoU

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True

train_losses = []
val_losses = []


import wandb

# Start a wandb run with `sync_tensorboard=True`
wandb.init(project="my-project", sync_tensorboard=True)

# plot and save metrics
def plot_metrics(metric_data, metric_name, epochs, save_path):
    """
    Plot metric data against epochs.
    Args:
        metric_data: List of metric values.
        metric_name: Name of the metric (e.g., Loss, mIoU).
        epochs: List of epoch numbers corresponding to the metric data.
        save_path: Path to save the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, metric_data, marker="o", label=metric_name)
    plt.xlabel("Epochs")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} over Epochs")
    plt.legend()
    plt.grid()
    plt.savefig(save_path)
    plt.close()

def parse_args():
    parser = ArgumentParser(description="Open-World Semantic Segmentation", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.set_common_args()
    args = parser.parse_args() # adds commonly used arguements like batch size, learning rate, etc
    if args.batch_size != 8:
        args.lr = args.lr * args.batch_size / 8 # adapt learning rate to batch size linearly
        print(f"Notice: adapting learning rate to {args.lr} because provided batch size differs from default batch size of 8.")
    return args

# set up the training environment, including directories for saving checkpoints and logs
def train_main():
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    args = parse_args()

    # directory for storing weights and other training related files
    training_starttime = datetime.now().strftime("%d_%m_%Y-%H_%M_%S-%f")
    ckpt_dir = os.path.join(args.results_dir, args.dataset, f"{args.id}", f"{training_starttime}")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(os.path.join(ckpt_dir, "confusion_matrices"), exist_ok=True)

    with open(os.path.join(ckpt_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    with open(os.path.join(ckpt_dir, "argsv.txt"), "w") as f:
        f.write(" ".join(sys.argv))
        f.write("\n")

    # =========================== data preparation ============================
    data_loaders = prepare_data(args, ckpt_dir)

    train_loader, valid_loader, _ = data_loaders

    n_classes_without_void = train_loader.dataset.n_classes_without_void
    if args.class_weighting != "None":
        class_weighting = train_loader.dataset.compute_class_weights(weight_mode=args.class_weighting, c=args.c_for_logarithmic_weighting)
    else:
        class_weighting = np.ones(n_classes_without_void)

    # ============================ model building =============================
    model, device = build_model(args, n_classes=n_classes_without_void)
    if args.freeze > 0:
        print("Freeze everything but the output layer(s).")
        for name, param in model.named_parameters():
            if "out" not in name:
                param.requires_grad = False

    # ============================ loss functions =============================
    loss_function_train = losses.CrossEntropyLoss2d(weight=class_weighting, device=device, void_label=-1) #void labels are ignored - include pixels that are do not have a valid label or in any way should be ignored
    loss_objectosphere = losses.ObjectosphereLoss(void_label=-1)
    loss_mav = losses.OWLoss(n_classes=n_classes_without_void, void_label=-1)
    loss_contrastive = losses.ContrastiveLoss(n_classes=n_classes_without_void, void_label=-1)

    pixel_sum_valid_data = valid_loader.dataset.compute_class_weights(weight_mode="linear")
    pixel_sum_valid_data_weighted = np.sum(pixel_sum_valid_data * class_weighting)
    loss_function_valid = losses.CrossEntropyLoss2dForValidData(
        weight=class_weighting, weighted_pixel_sum=pixel_sum_valid_data_weighted, device=device, void_label=-1
    )

    train_loss = [loss_function_train, loss_objectosphere, loss_mav, loss_contrastive]
    val_loss = [loss_function_valid, loss_objectosphere, loss_mav, loss_contrastive]
    if not args.obj:
        train_loss[1] = None
        val_loss[1] = None
    if not args.mav:
        train_loss[2] = None
        val_loss[2] = None
    if not args.closs:
        train_loss[3] = None
        val_loss[3] = None

    optimizer = get_optimizer(args, model)

    # ============================= lr scheduler ==============================
    lr_scheduler = OneCycleLR(
        optimizer,
        max_lr=[i["lr"] for i in optimizer.param_groups],
        total_steps=args.epochs,
        div_factor=25,
        pct_start=0.1,
        anneal_strategy="cos",
        final_div_factor=1e4,
    )

    # =========== load checkpoint if parameter last_ckpt is provided ==========
    if args.last_ckpt:
        ckpt_path = args.last_ckpt
        epoch_last_ckpt, best_miou, best_miou_epoch, mav_dict, std_dict = load_ckpt(model, optimizer, ckpt_path, device)
        start_epoch = epoch_last_ckpt + 1
    else:
        start_epoch = 0
        best_miou = 0
        best_miou_epoch = 0

    if args.load_weights:
        model.load_state_dict(torch.load(args.load_weights))

    writer = SummaryWriter("runs/" + ckpt_dir.split(args.dataset)[-1])

    # ============================= start training ============================
    for epoch in range(int(start_epoch), args.epochs):
        # unfreeze
        if args.freeze == epoch and args.finetune is None:
            for param in model.parameters():
                param.requires_grad = True

        mean, var, total_train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            device=device,
            optimizer=optimizer,
            train_loss=train_loss,
            epoch=epoch,
            lr_scheduler=lr_scheduler,
            debug_mode=args.debug,
            writer=writer,
        )

        # TODO: IMPLEMENT VALIDATION plot_results=True: FOR VIZUALIZATION OF RESULTS
        # TODO: PLOT TRAINING+VALIDATION LOSSES WITH WANDB OR TENSORBOARD
        # TODO: The validate function to visualize the results
        
        print("VALIDATION TO BE IMPLEMENTED")
        miou = validate(
            model=model,
            valid_loader=valid_loader,
            device=device,
            val_loss=val_loss,
            epoch=epoch,
            debug_mode=args.debug,
            writer=writer,
            classes=args.num_classes,
            plot_results=False,
            plot_path= os.path.join(ckpt_dir, "val_results")
        )

        writer.flush()
        wandb.finish()

        # Track metrics for plotting
        print(total_train_loss)
        if total_train_loss < 200: # TO REMOVE HIGH SPIKES IN LOSS - REMOVE LATER
            train_losses.append(total_train_loss)

        val_losses.append(miou) # Example: track validation mIoU

        # Plot metrics every 20 epochs
        if (epoch + 1) % 20 == 0:
            epochs = list(range(1, len(train_losses) + 1))

            # Plot training loss
            plot_metrics(train_losses, "Training Loss", epochs, f"{ckpt_dir}/train_loss_epoch_{epoch + 1}.png")
            plot_metrics(val_losses, "Validation mIoU", epochs, f"{ckpt_dir}/val_mIoU_epoch_{epoch + 1}.png")

        # Save weights
        if not args.overfit:
            save_ckpt_every_epoch(ckpt_dir, model, optimizer, epoch, best_miou, best_miou_epoch, mean, var)
            if (epoch + 1) % 20 == 0:
                torch.save(model.state_dict(), os.path.join(ckpt_dir, f"epoch_{epoch + 1}.pth"))
            if miou > best_miou:
                best_miou = miou
                best_miou_epoch = epoch
                torch.save(model.state_dict(), os.path.join(ckpt_dir, "best_miou.pth"))

    print("Training completed ")


def train_one_epoch(
    model,
    train_loader,
    device,
    optimizer,
    train_loss,
    epoch,
    lr_scheduler,
    writer,
    debug_mode=False,
):
    samples_of_epoch = 0

    # set model to train mode
    model.train()

    loss_function_train, loss_obj, loss_mav, loss_contrastive = train_loss

    # summed loss of all resolutions
    total_loss_list = []
    total_sem_loss = []
    total_obj_loss = []
    total_ows_loss = []
    total_con_loss = []

    mavs = None
    if epoch and loss_contrastive is not None:
        mavs = loss_mav.read()
    for i, sample in enumerate(train_loader):
        start_time_for_one_step = time.time()

        # load the data and send them to gpu
        image = sample["image"].to(device)
        batch_size = image.data.shape[0]

        label = sample["label"].clone().long().to(device)
        # label[label == 255] = -1

        for param in model.parameters():
            param.grad = None

        # forward pass
        pred_scales, ow_res = model(image)
        # cw_target[cw_target > 16] = 255
        # for some reason we need clone here, not idea why
        losses = loss_function_train(pred_scales, label.clone())
        loss_segmentation = sum(losses)
        loss_objectosphere = torch.tensor(0.0)
        loss_ows = torch.tensor(0.0)
        loss_con = torch.tensor(0.0)
        total_loss = 0.9 * loss_segmentation
        # label[label < 0] = -1

        if loss_obj is not None:
            loss_objectosphere = loss_obj(ow_res, label)
            total_loss += 0.5 * loss_objectosphere
        if loss_mav is not None:
            loss_ows = loss_mav(pred_scales, label, is_train=True)
            total_loss += 0.1 * loss_ows
        if loss_contrastive is not None:
            loss_con = loss_contrastive(mavs, ow_res, label, epoch)
            total_loss += 0.5 * loss_con

        total_loss.backward()
        optimizer.step()

        # append loss values to the lists. Later we can calculate the
        # mean training loss of this epoch
        total_loss = total_loss.detach().cpu().numpy()
        loss_segmentation = loss_segmentation.detach().cpu().numpy()
        loss_objectosphere = loss_objectosphere.detach().cpu().numpy()
        loss_ows = loss_ows.detach().cpu().numpy()
        loss_con = loss_con.detach().cpu().numpy()

        total_loss_list.append(total_loss)
        total_sem_loss.append(loss_segmentation)
        total_obj_loss.append(loss_objectosphere)
        total_ows_loss.append(loss_ows)
        total_con_loss.append(loss_con)

        if np.isnan(total_loss):
            import ipdb;ipdb.set_trace()  # fmt: skip
            raise ValueError("Loss is None")

        # print log
        samples_of_epoch += batch_size
        time_inter = time.time() - start_time_for_one_step

        learning_rates = lr_scheduler.get_last_lr()

        print_log(
            epoch,
            samples_of_epoch,
            batch_size,
            len(train_loader.dataset),
            total_loss,
            time_inter,
            learning_rates,
        )

        if debug_mode:
            # only one batch while debugging
            break

        # break #### REMOVE LATER

    # fill the logs for csv log file and web logger
    writer.add_scalar("Loss/train", np.mean(total_loss_list), epoch)
    writer.add_scalar("Loss/semantic", np.mean(total_sem_loss), epoch)
    writer.add_scalar("Loss/objectosphere", np.mean(total_obj_loss), epoch)
    writer.add_scalar("Loss/ows", np.mean(total_ows_loss), epoch)
    writer.add_scalar("Loss/contrastive", np.mean(total_con_loss), epoch)

    lr_scheduler.step()

    if loss_mav is not None:
        mean, var = loss_mav.update()
        return mean, var, np.mean(total_loss_list)
    else:
        return {}, {}, {}

def validate(
    model,
    valid_loader,
    device,
    val_loss,
    epoch,
    writer,
    loss_function_valid_unweighted=None,
    add_log_key="",
    debug_mode=False,
    classes=19,
    plot_results=False,
    plot_path=None,
):
    # set model to eval mode
    model.eval()

    # we want to store miou and ious for each camera
    miou = dict()
    ious = dict()

    loss_function_valid, loss_obj, loss_mav, loss_contrastive = val_loss

    # reset loss (of last validation) to zero
    loss_function_valid.reset_loss()

    if loss_function_valid_unweighted is not None:
        loss_function_valid_unweighted.reset_loss()

    compute_iou = IoU(task="multiclass", num_classes=classes, average="none", ignore_index=-1).to(device)

    mavs = None
    if loss_contrastive is not None:
        mavs = loss_mav.read()

    if plot_results:
        with open("mavs.pickle", "rb") as h1:
            mavs = pickle.load(h1)
        with open("vars.pickle", "rb") as h2:
            vars = pickle.load(h2)
        mavs = torch.vstack(tuple(mavs.values())).detach().to("cpu") # 19x19

    total_loss_obj = []
    total_loss_mav = []
    total_loss_con = []
    # validate each camera after another as all images of one camera have
    # the same resolution and can be resized together to the ground truth
    # segmentation size.

    for i, sample in enumerate(tqdm(valid_loader, desc="Validation")):
        # copy the data to gpu
        image = sample["image"].to(device)
        target = sample["label"].clone().long().to(device)

        if not device.type == "cpu":
            torch.cuda.synchronize()

        # forward pass
        with torch.no_grad():
            prediction_ss, prediction_ow = model(image)

            if plot_results and i < 1:  # Limit to first 8 samples for visualization
                # Create figure with 8 rows and 4 columns (adding a column for prediction_ow)
                fig, axes = plt.subplots(9, 5, figsize=(16, 24))  # One extra row for column names

                # Add column names in the first row
                column_names = ["Image", "Prediction (SS)", "Prediction (OW)", "Ground Truth", "OW Binary GT"]
                for col_idx, col_name in enumerate(column_names):
                    axes[0, col_idx].text(
                        0.5, 0.5, col_name, ha="center", va="center", fontsize=16, weight="bold"
                    )
                    axes[0, col_idx].axis("off")  # Turn off axes for header

                # Add images in the subsequent rows
                for idx in range(8):  # Limit to 8 rows of images
                    if idx >= len(sample["image"]):  # Skip if less than 8 images in batch
                        break

                    # Normalize and process the input image
                    image_np = image[idx].detach().cpu().permute(1, 2, 0).numpy()
                    image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())

                    # Process prediction_ss (semantic segmentation prediction)
                    pred_ss_np = torch.argmax(torch.softmax(prediction_ss[idx], dim=0), dim=0).detach().cpu().numpy()

                    # Process prediction_ow (open-world segmentation prediction)
                    # pred_ow_np = prediction_ow[idx, 0].detach().cpu().numpy()
                    ows_target = target.long() - 1
                    ows_target[ows_target < classes] = 0
                    ows_binary_gt = 255*ows_target.bool().long()
                    ows_binary_gt = ows_binary_gt[idx].detach().cpu().numpy()
                    s_cont = contrastive_inference(prediction_ow)
                    s_sem, similarity = semantic_inference(prediction_ss, mavs, vars)
                    s_sem = s_sem.to("cuda")
                    s_unk = (s_cont + s_sem) / 2
                    pred_ow_np = 255*(s_unk - 0.6).relu().bool().int()
                    pred_ow_np = pred_ow_np[idx].detach().cpu().numpy()

                    # Process the ground truth
                    target_np = target[idx].detach().cpu().numpy()

                    # Display Image, Prediction (SS), Prediction (OW), and Ground Truth
                    axes[idx + 1, 0].imshow(image_np)
                    axes[idx + 1, 0].axis("off")

                    axes[idx + 1, 1].imshow(pred_ss_np)
                    axes[idx + 1, 1].axis("off")

                    axes[idx + 1, 2].imshow(pred_ow_np)
                    axes[idx + 1, 2].axis("off")

                    axes[idx + 1, 3].imshow(target_np)
                    axes[idx + 1, 3].axis("off")

                    axes[idx + 1, 4].imshow(ows_binary_gt)
                    axes[idx + 1, 4].axis("off")

                # Save the plot
                plot_dir = plot_path  # Ensure this is the directory path
                os.makedirs(plot_dir, exist_ok=True)  # Create the directory if it doesn't exist
                plot_file = f"val_imgs_epoch_{epoch}.png"
                full_plot_path = os.path.join(plot_dir, plot_file)  # Full path for the file

                # Save the plot
                plt.savefig(full_plot_path, bbox_inches="tight")
                plt.close(fig)

            compute_iou.update(prediction_ss, target)
            
            # compute valid loss
            loss_function_valid.add_loss_of_batch(prediction_ss, target.clone())

            loss_objectosphere = torch.tensor(0)
            loss_ows = torch.tensor(0)
            loss_con = torch.tensor(0)
            if loss_obj is not None:
                ###???
                # target_obj = sample["label"]
                # target_obj[target_obj == 16] = -1
                # target_obj[target_obj == 17] = -1
                # target_obj[target_obj == 18] = -1
                ###???
                loss_objectosphere = loss_obj(prediction_ow, target)
            total_loss_obj.append(loss_objectosphere.detach().cpu().numpy())
            if loss_mav is not None:
                loss_ows = loss_mav(prediction_ss, target, is_train=False)
            total_loss_mav.append(loss_ows.detach().cpu().numpy())
            if loss_contrastive is not None:
                loss_con = loss_contrastive(mavs, prediction_ow, target, epoch)
            total_loss_con.append(loss_con.detach().cpu().numpy())

            if debug_mode:
                # only one batch while debugging
                break
    
    ious = compute_iou.compute().detach().cpu()
    miou = ious.mean()

    print("mIoU at epoch {}: {}".format(epoch, miou))

    total_loss = (
        loss_function_valid.compute_whole_loss()
        + np.mean(total_loss_obj)
        + np.mean(total_loss_mav)
        + np.mean(total_loss_con)
    )
    
    # fill the logs for csv log file and web logger
    writer.add_scalar("Loss/val", total_loss, epoch)
    writer.add_scalar("Metrics/miou", miou, epoch)
    for i, iou in enumerate(ious):
        writer.add_scalar(
            "Class_metrics/iou_{}".format(i),
            torch.mean(iou),
            epoch,
        )
    return miou


def test_ow(
    model,
    test_loader,
    device,
    val_loss,
    epoch,
    writer,
    classes=19,
    mean=None,
    var=None,
):
    delta = 0.6

    # set model to eval mode
    model.eval()

    compute_iou = IoU(task="multiclass", num_classes=2, average="none", ignore_index=-1).to(device)

    _, loss_obj, loss_mav, _ = val_loss

    with open("mavs.pickle", "rb") as h1:
        mavs = pickle.load(h1)
    with open("vars.pickle", "rb") as h2:
        vars = pickle.load(h2)

    mavs = torch.vstack(tuple(mavs.values())).detach().cpu()()  # 19x19
    new_mavs = None

    for i, sample in enumerate(tqdm(test_loader, desc="Testing")):
        # copy the data to gpu
        image = sample["image"].to(device)
        label = sample["label"].long().to(device)

        if not device.type == "cpu":
            torch.cuda.synchronize()

        # forward pass
        with torch.no_grad():
            prediction, ow_pred = model(image)

            ows_target = label.long() - 1
            ows_target[ows_target < classes] = 0
            ows_binary_gt = ows_target.bool().long()

            s_cont = contrastive_inference(ow_pred)
            s_sem, similarity = semantic_inference(prediction, mavs, vars)
            s_unk = (s_cont + s_sem) / 2

            ows_binary_pred = (s_unk - delta).relu().bool().int()

            compute_iou.update(ows_binary_pred, ows_binary_gt)

            prediction = prediction.permute(1, 0, 2, 3)
            unk_pixels = prediction[:, :, ows_binary_pred == 0]

            tmp = torch.ones(unk_pixels.shape)
            if new_mavs is not None:
                for i in range(new_mavs.shape[0]):
                    mav = new_mavs[:, i].unsqueeze(1)
                    dist = torch.norm(unk_pixels - mav, dim=0)
                    dist = (dist < 0.5).int()
                    tmp[:, dist == 1] = 0
                    upd = torch.mean(unk_pixels[dist == 1], dim=0)
                    new_mavs[i, :] = (new_mavs[i, :] + upd) / 2
            preds = unk_pixels * tmp
            preds = torch.unique(preds, dim=1)
            if tmp.sum():
                preds = preds[:, 1:]

            clusters = ac(n_clusters=None, affinity="euclidean", distance_threshold=0.5).fit(preds.detach().cpu()().numpy().T)
            groups = clusters.labels_

            nc = groups.max()
            for c in nc:
                new = preds[:, groups == c]
                new = torch.mean(torch.tensor(new), dim=1)
                if new_mavs is None:
                    new_mavs = new
                else:
                    new_mavs = torch.vstack((new_mavs, new))

    ious = compute_iou.compute().detach().cpu()
    writer.add_scalar("Metrics/OWS/known", ious[0], epoch)
    writer.add_scalar("Metrics/OWS/unknown", ious[1], epoch)


def contrastive_inference(predictions, radius=1.0):
    scores = torch.nn.functional.relu(1 - torch.norm(predictions, dim=1) / radius)
    return scores


def semantic_inference(predictions, mavs, var):
    mavs = mavs.to("cuda")
    predictions = predictions.to("cuda")
    stds = torch.vstack(tuple(var.values()))
    # stds = torch.vstack(tuple(var.values())).detach().cpu()  # 19x19
    d_pred = (predictions[:, None, ...] - mavs[None, :, :, None, None])  # [8,1,19,h,w] - [1,19,19,1,1]
    d_pred_ = d_pred / (stds[None, :, :, None, None] + 1e-8)
    scores = torch.exp(-torch.einsum("bcfhw,bcfhw->bchw", d_pred_, d_pred) / 2)
    best = scores.max(dim=1)
    return 1 - best[0], best[1]


def get_optimizer(args, model):
    # set different learning rates fo different parts of the model
    # when using default parameters the whole model is trained with the same
    # learning rate
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
            nesterov=True,
        )
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999),
        )
    else:
        raise NotImplementedError("Currently only SGD and Adam as optimizers are supported. Got {}".format(args.optimizer))

    print("Using {} as optimizer".format(args.optimizer))
    print("\n\n=========================================================================\n\n")
    return optimizer


if __name__ == "__main__":
    train_main()
