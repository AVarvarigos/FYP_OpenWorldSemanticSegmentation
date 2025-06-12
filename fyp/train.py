########################################################
#                                                      #
#       author: omitted for anonymous submission       #
#                                                      #
#     credits and copyright coming upon publication    #
#                                                      #
########################################################

import argparse
import json
import os
import pickle
import sys
from datetime import datetime
import random
import numpy as np
import torch
import torch.optim
from sklearn.cluster import AgglomerativeClustering as ac
from src import losses
from src.args import ArgumentParser
from src.build_model import build_model
from src.prepare_data import prepare_data
from src.utils import load_ckpt, save_ckpt_every_epoch
from src.plot_images import plot_images
from torch.functional import F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import JaccardIndex as IoU
from torchmetrics import Recall, Precision, F1Score, AveragePrecision
from tqdm import tqdm

import wandb

# Start a wandb run with `sync_tensorboard=True`
wandb.init(project="final-runs-fyp", sync_tensorboard=True)


def parse_args():
    parser = ArgumentParser(
        description="Open-World Semantic Segmentation (Training)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.set_common_args()
    args = parser.parse_args()
    # The provided learning rate refers to the default batch size of 8.
    # When using different batch sizes we need to adjust the learning rate
    # accordingly:
    args.batch_size_valid = args.batch_size
    if args.batch_size != 8:
        args.lr = args.lr * args.batch_size / 8
        print(
            f"Notice: adapting learning rate to {args.lr} because provided "
            f"batch size differs from default batch size of 8."
        )

    return args


def train_main():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    args = parse_args()

    print(args)

    # directory for storing weights and other training related files
    training_starttime = datetime.now().strftime("%d_%m_%Y-%H_%M_%S-%f")
    ckpt_dir = os.path.join(
        args.results_dir, args.dataset, f"{args.id}", f"{training_starttime}"
    )
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(os.path.join(ckpt_dir, "confusion_matrices"), exist_ok=True)

    with open(os.path.join(ckpt_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    with open(os.path.join(ckpt_dir, "argsv.txt"), "w") as f:
        f.write(" ".join(sys.argv))
        f.write("\n")

    # data preparation ---------------------------------------------------------
    data_loaders = prepare_data(args, ckpt_dir)

    train_loader, valid_loader, _ = data_loaders

    n_classes_without_void = train_loader.dataset.n_classes_without_void
    if args.class_weighting != "None":
        class_weighting = train_loader.dataset.compute_class_weights(
            weight_mode=args.class_weighting, c=args.c_for_logarithmic_weighting
        )
    else:
        class_weighting = np.ones(n_classes_without_void)
    # model building -----------------------------------------------------------
    model, device = build_model(args, n_classes=n_classes_without_void)
    
    # Print initial model parameter info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = count_trainable_parameters(model)
    print(f"\n📊 MODEL PARAMETERS:")
    print(f"   Total: {total_params/1_000_000:.2f}M parameters")
    print(f"   Trainable: {trainable_params/1_000_000:.2f}M parameters ({100*trainable_params/total_params:.1f}%)")
    
    if args.freeze > 0:
        print("Freeze everything but the output layer(s).")
        for name, param in model.named_parameters():
            if "out" not in name:
                param.requires_grad = False

        # Print parameters after freezing
        trainable_after_freeze = count_trainable_parameters(model)
        print(f"   After freezing: {trainable_after_freeze/1_000_000:.2f}M trainable parameters ({100*trainable_after_freeze/total_params:.1f}%)")
        print(f"   Will unfreeze at epoch {args.freeze}")
    print()

    # loss, optimizer, learning rate scheduler, csvlogger  ----------

    # loss functions
    loss_function_train = losses.CrossEntropyLoss2d(weight=class_weighting, device=device)
    focal_loss = losses.FocalLoss(alpha=class_weighting)
    dice_loss = losses.DiceLoss()
    loss_objectosphere = losses.ObjectosphereLoss()
    loss_mav = losses.OWLoss(n_classes=n_classes_without_void)
    loss_contrastive = losses.ContrastiveLoss(n_classes=n_classes_without_void)
    loss_function_valid = losses.CrossEntropyLoss2dForValidData(
        weight=class_weighting,
        device=device,
    )

    train_loss = [loss_function_train, loss_objectosphere, loss_mav, loss_contrastive, focal_loss, dice_loss]
    val_loss = [loss_function_valid, loss_objectosphere, loss_mav, loss_contrastive, focal_loss, dice_loss]
    if not args.obj:
        train_loss[1] = None
        val_loss[1] = None
    if not args.mav:
        train_loss[2] = None
        val_loss[2] = None
    if not args.closs:
        train_loss[3] = None
        val_loss[3] = None
    if not args.focal:
        train_loss[4] = None
        val_loss[4] = None
    if not args.dice:
        train_loss[5] = None
        val_loss[5] = None

    optimizer = get_optimizer(args, model)

    lr_scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
    )

    # load checkpoint if parameter last_ckpt is provided
    if args.last_ckpt:
        ckpt_path = args.last_ckpt
        epoch_last_ckpt, best_miou, best_miou_epoch, mav_dict, std_dict, ows_loss = load_ckpt(
            model, optimizer, ckpt_path, device
        )
        start_epoch = epoch_last_ckpt + 1
        train_loss[2] = ows_loss
        val_loss[2] = ows_loss
    else:
        start_epoch = 0
        best_miou = 0
        best_miou_epoch = 0

    if args.load_weights:
        model.load_state_dict(torch.load(args.load_weights))

    writer = SummaryWriter("runs/" + ckpt_dir.split(args.dataset)[-1])

    # start training -----------------------------------------------------------
    for epoch in range(int(start_epoch), args.epochs):
        mean, var = train_one_epoch(
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

        # validate every 3 epochs
        if args.overfit and epoch % 3 != 0:
            continue

        miou = validate(
            model=model,
            var=var,
            valid_loader=valid_loader,
            device=device,
            val_loss=val_loss,
            epoch=epoch,
            debug_mode=args.debug,
            writer=writer,
            classes=args.num_classes,
            plot_results=args.plot_results,
            plot_path=os.path.join(ckpt_dir, "val_results")
        )

        writer.flush()

        # save weights
        if not args.overfit:
            # save / overwrite latest weights (useful for resuming training)
            save_ckpt_every_epoch(
                ckpt_dir, model, optimizer, epoch, best_miou, best_miou_epoch, mean, var, train_loss[2]
            )
            if (epoch + 1) % 20 == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(ckpt_dir, "epoch_" + str(epoch) + ".pth"),
                )
            if miou > best_miou:
                best_miou = miou
                best_miou_epoch = epoch
                torch.save(
                    model.state_dict(),
                    os.path.join(ckpt_dir, "best_miou.pth"),
                )

    # save mavs to a pickle
    wandb.finish()
    with open("mavs.pickle", "wb") as h1:
        pickle.dump(mean, h1, protocol=pickle.HIGHEST_PROTOCOL)
    with open("vars.pickle", "wb") as h2:
        pickle.dump(var, h2, protocol=pickle.HIGHEST_PROTOCOL)

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

    loss_function_train, loss_obj, loss_mav, loss_contrastive, loss_focal, loss_dice = train_loss

    # summed loss of all resolutions
    losses_lists = {
        loss_name: []
        for loss_name in ["total", "segmentation", "objectsphere", "ows", "contrastive", "dice", "focal"]
    }
    loss_weights = {
        "segmentation": 1.0,
        "objectsphere": 0.5,
        "ows": 0.1,
        "contrastive": 0.5,
        "dice": 0.5,
        "focal": 3.0,
    }

    mavs = None
    if epoch and loss_contrastive is not None:
        mavs = loss_mav.read()

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for sample in progress_bar:
        # load the data and send them to gpu
        image = sample["image"].to(device)
        batch_size = image.data.shape[0]

        # Due to how preprocessing works
        label = sample["label"].cuda().long() - 2

        # NOTE: for future me, for some weird reason I can not explain, this is necessary.
        for param in model.parameters():
            param.grad = None

        # forward pass
        pred_scales, ow_res = model(image)
        loss = loss_function_train(pred_scales, label.clone().cuda())

        losses = {
            "segmentation": loss,
            "objectsphere": loss_obj,
            "ows": loss_mav,
            "contrastive": loss_contrastive,
            "dice": loss_dice,
            "focal": loss_focal
        }

        if losses["focal"] is not None:
            losses["focal"] = losses["focal"](pred_scales, label.clone().cuda())
            losses.pop("segmentation", None)  # focal loss replaces segmentation loss
        if losses["dice"] is not None:
            losses["dice"] = losses["dice"](pred_scales, label.clone().cuda())
        if losses["objectsphere"] is not None:
            losses["objectsphere"] = losses["objectsphere"](ow_res, label.clone().cuda())
        if losses["ows"] is not None:
            losses["ows"] = losses["ows"](pred_scales, label.clone().cuda(), is_train=True)
        if losses["contrastive"] is not None:
            losses["contrastive"] = losses["contrastive"](mavs, ow_res, label.clone().cuda(), epoch)

        losses = {
            loss_name: loss_item
            for loss_name, loss_item in losses.items()
            if loss_item is not None
        }

        total_loss = 0
        for loss_name in losses:
            total_loss += loss_weights[loss_name] * losses[loss_name]
        losses["total"] = total_loss

        for loss_name, loss_i in losses.items():
            losses_lists[loss_name].append(loss_i.detach())

        if torch.isnan(losses["total"]):
            import ipdb;ipdb.set_trace()  # fmt: skip
            raise ValueError("Loss is None")

        total_loss.backward()
        optimizer.step()

        # print log
        samples_of_epoch += batch_size

        learning_rates = [
            pg['lr'] for pg in optimizer.param_groups
        ]

        # Update progress bar
        progress_bar.set_postfix({
            'batch_loss': f'{losses["total"]:.4f}',
            'lr': f'{learning_rates[0]:.6f}'
        })

        if debug_mode:
            # only one batch while debugging
            break

    lr_scheduler.step(torch.tensor(losses_lists["total"]).mean().item())

    # fill the logs for csv log file and web logger
    for loss_name, loss_list in losses_lists.items():
        if not loss_list:
            continue
        writer.add_scalar(f"Loss/{loss_name}", torch.tensor(loss_list).mean().item(), epoch)

    if loss_mav is not None:
        mean, var = loss_mav.update()
        return mean, var
    else:
        return {}, {}


def validate(
    model,
    var,
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
    is_train_loader = False
):
    valid_split = valid_loader.dataset.split + add_log_key

    # we want to track how long each part of the validation takes
    forward_time = 0
    copy_to_gpu_time = 0

    # set model to eval mode
    model.eval()

    # we want to store miou and ious for each camera
    miou = dict()
    ious = dict()

    loss_function_valid, loss_obj, loss_mav, loss_contrastive, loss_focal, loss_dice = val_loss

    # reset loss (of last validation) to zero
    loss_function_valid.reset_loss()

    if loss_function_valid_unweighted is not None:
        loss_function_valid_unweighted.reset_loss()

    compute_iou = IoU(
        task="multiclass", num_classes=classes, average="none", ignore_index=255
    ).to(device)
    compute_recall = Recall(
        task="multiclass", num_classes=classes, average="none", ignore_index=255, top_k=1
    ).to(device)
    compute_precision = Precision(
        task="multiclass", num_classes=classes, average="none", ignore_index=255, top_k=1
    ).to(device)
    compute_f1 = F1Score(
        task="multiclass", num_classes=classes, average="none", ignore_index=255, top_k=1
    ).to(device)
    aupr_list = []

    mavs = None
    if loss_contrastive is not None:
        mavs = loss_mav.read()

    total_loss_obj = []
    total_loss_mav = []
    total_loss_con = []
    total_loss_focal = []
    total_loss_dice = []
    # validate each camera after another as all images of one camera have
    # the same resolution and can be resized together to the ground truth
    # segmentation size.

    desc = "Valid step"
    if is_train_loader:
        desc = "Train Validation step"
    for i, sample in enumerate(tqdm(valid_loader, desc=desc)):
        # copy the data to gpu
        image = sample["image"].to(device)

        if not device.type == "cpu":
            torch.cuda.synchronize()

        # forward pass
        with torch.no_grad():
            prediction_ss, prediction_ow = model(image)

            if not device.type == "cpu":
                torch.cuda.synchronize()

            compute_auprc = AveragePrecision(
                task="multiclass", num_classes=classes, average="none", ignore_index=255
            ).to(device)

            target = sample["label"].long().to(device) - 2
            # target = sample["label"].clone().cuda()
            target_iou = target.clone()
            target_iou[target_iou == -1] = 255
            compute_iou.update(prediction_ss, target_iou)
            compute_recall.update(prediction_ss, target_iou)
            compute_precision.update(prediction_ss, target_iou)
            compute_f1.update(prediction_ss, target_iou)
            compute_auprc.update(prediction_ss, target_iou)
            auprc = compute_auprc(prediction_ss, target_iou)
            np.nan_to_num(auprc, copy=False)  # replace NaN with 0
            aupr_list.append(auprc.cpu().detach().numpy())

            # plot images every 3 epochs
            if epoch % 3 == 0 and i == 0 and plot_results:
                plot_images(epoch, sample, image, mavs, var, prediction_ss, prediction_ow, target, classes, use_mav=mavs is not None, plot_path=plot_path)

            # compute valid loss
            loss_function_valid.add_loss_of_batch(prediction_ss, target)

            loss_objectosphere = torch.tensor(0)
            loss_ows = torch.tensor(0)
            loss_con = torch.tensor(0)
            i_loss_dice = torch.tensor(0)
            i_loss_focal = torch.tensor(0)
            if loss_focal is not None:
                i_loss_focal = loss_focal(prediction_ss, target)
            total_loss_focal.append(i_loss_focal.cpu().detach().numpy())
            if loss_dice is not None:
                i_loss_dice = loss_dice(prediction_ss, target)
            total_loss_dice.append(i_loss_dice.cpu().detach().numpy())
            if loss_obj is not None:
                loss_objectosphere = loss_obj(prediction_ow, target)
            total_loss_obj.append(loss_objectosphere.cpu().detach().numpy())
            if loss_mav is not None:
                loss_ows = loss_mav(prediction_ss, target, is_train=False)
            total_loss_mav.append(loss_ows.cpu().detach().numpy())
            if loss_contrastive is not None:
                loss_con = loss_contrastive(mavs, prediction_ow, target, epoch)
            total_loss_con.append(loss_con.cpu().detach().numpy())

            if debug_mode:
                # only one batch while debugging
                break

    ious = compute_iou.compute().detach().cpu()
    miou = ious.mean()

    recall = compute_recall.compute().detach().cpu()
    m_recall = recall.mean()

    precision = compute_precision.compute().detach().cpu()
    m_precision = precision.mean()

    f1 = compute_f1.compute().detach().cpu()
    m_f1 = f1.mean()

    auprc = np.array(aupr_list).mean(axis=0)
    m_auprc = np.mean(auprc)

    total_loss = (
        loss_function_valid.compute_whole_loss()
        + np.mean(total_loss_obj)
        + np.mean(total_loss_mav)
        + np.mean(total_loss_con)
        + np.mean(total_loss_focal)
        + np.mean(total_loss_dice)
    )
    prefix = "Train" if is_train_loader else "Valid"
    writer.add_scalar(f"{prefix}/Loss", total_loss, epoch)
    writer.add_scalar(f"{prefix}/miou", miou, epoch)
    writer.add_scalar(f"{prefix}/average_recall", m_recall, epoch)
    writer.add_scalar(f"{prefix}/average_precision", m_precision, epoch)
    writer.add_scalar(f"{prefix}/average_f1", m_f1, epoch)
    writer.add_scalar(f"{prefix}/average_auprc", m_auprc, epoch)
    for i, iou in enumerate(ious):
        writer.add_scalar(
            f"{prefix}/Class_metrics/iou_{i}",
            torch.mean(iou),
            epoch,
        )
        writer.add_scalar(
            f"{prefix}/Class_metrics/recall_{i}",
            torch.mean(recall[i]),
            epoch,
        )
        writer.add_scalar(
            f"{prefix}/Class_metrics/precision_{i}",
            torch.mean(precision[i]),
            epoch,
        )
        writer.add_scalar(
            f"{prefix}/Class_metrics/f1_{i}",
            torch.mean(f1[i]),
            epoch
        )
        writer.add_scalar(
            f"{prefix}/Class_metrics/auprc_{i}",
            torch.mean(auprc[i]),
            epoch
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

    compute_iou = IoU(
        task="multiclass", num_classes=2, average="none", ignore_index=255
    ).to(device)

    _, loss_obj, loss_mav, _ = val_loss

    with open("mavs.pickle", "rb") as h1:
        mavs = pickle.load(h1)
    with open("vars.pickle", "rb") as h2:
        vars = pickle.load(h2)

    mavs = torch.vstack(tuple(mavs.values())).cpu()  # 19x19
    new_mavs = None

    for i, sample in enumerate(tqdm(test_loader, desc="Test step")):
        # copy the data to gpu
        image = sample["image"].to(device)
        label = sample["label"].to(device) - 1

        if not device.type == "cpu":
            torch.cuda.synchronize()

        # forward pass
        with torch.no_grad():
            prediction, ow_pred = model(image)

            ows_target = label.long() - 2
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

            clusters = ac(
                n_clusters=None, affinity="euclidean", distance_threshold=0.5
            ).fit(preds.cpu().numpy().T)
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

def count_trainable_parameters(model):
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_trainable_params(model, epoch):
    """Print trainable parameters in millions for the current epoch."""
    # Shows: Epoch X | Trainable: Y.YYM/Z.ZZM params (XX.X%)

if __name__ == "__main__":
    train_main()
