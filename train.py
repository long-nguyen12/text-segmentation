from utils.data import *
from utils.metric import *
from utils.lr_scheduler import *
from utils.utils import *
from argparse import ArgumentParser
import torch
from model.net import *
from model.loss import *
from tqdm import tqdm
import os.path as osp
import os
import time
from glob import glob
import albumentations as A
import cv2


def parse_args():
    parser = ArgumentParser(description="Implement of model")

    parser.add_argument("--train_path", type=str, default="data/IRSTD-1k")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.05)

    parser.add_argument("--base-size", type=int, default=256)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--warm-epoch", type=int, default=5)

    parser.add_argument("--mode", type=str, default="train")
    args = parser.parse_args()
    return args


class Dataset(torch.utils.data.Dataset):

    def __init__(self, img_paths, mask_paths, transform=None, mask_transform=None):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, 0)

        if self.transform is not None:
            img_aug = self.transform(image=image)
            image = img_aug["image"]

        if self.mask_transform is not None:
            mask_aug = self.mask_transform(image=mask)
            mask = mask_aug["image"]

        image = image.astype("float32") / 255
        image = image.transpose((2, 0, 1))

        mask = mask[:, :, np.newaxis]
        mask = mask.astype("float32") / 255
        mask = mask.transpose((2, 0, 1))
        return np.asarray(image), np.asarray(mask)


epsilon = 1e-7


def recall_m(y_true, y_pred):
    true_positives = torch.sum(torch.round(torch.clip(y_true * y_pred, 0, 1)))
    possible_positives = torch.sum(torch.round(torch.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + epsilon)
    return recall


def precision_m(y_true, y_pred):
    true_positives = torch.sum(torch.round(torch.clip(y_true * y_pred, 0, 1)))
    predicted_positives = torch.sum(torch.round(torch.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + epsilon)
    return precision


def dice_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + epsilon))


def iou_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return recall * precision / (recall + precision - recall * precision + epsilon)


class Trainer(object):
    def __init__(self, args):
        assert args.mode == "train" or args.mode == "test"

        self.args = args
        self.start_epoch = 0
        self.mode = args.mode

        dataset = args.train_path.split("/")[-1]
        self.save_folder = f"snapshots/{dataset}"

        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder, exist_ok=True)
        else:
            print("Save path existed")

        train_img_paths = []
        train_mask_paths = []
        train_img_paths = glob("{}/images/Train/*".format(args.train_path))
        train_mask_paths = glob("{}/masks/Train/*".format(args.train_path))
        train_img_paths.sort()
        train_mask_paths.sort()

        transform = A.Compose(
            [
                A.Resize(height=256, width=256),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

        mask_transform = A.Compose(
            [
                A.Resize(height=256, width=256),
            ]
        )

        train_dataset = Dataset(
            train_img_paths,
            train_mask_paths,
            transform=transform,
            mask_transform=mask_transform,
        )

        val_img_paths = []
        val_mask_paths = []
        val_img_paths = glob("{}/images/Test/*".format(args.train_path))
        val_mask_paths = glob("{}/masks/Test/*".format(args.train_path))
        val_img_paths.sort()
        val_mask_paths.sort()

        val_dataset = Dataset(
            val_img_paths,
            val_mask_paths,
            transform=transform,
            mask_transform=mask_transform,
        )

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batchsize,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

        self.val_loader = Dataset(
            val_dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
        )

        device = torch.device("cuda")
        self.device = device

        # Model
        model = SegmentNet()
        model.to(device)
        self.model = model

        # Optimizer and Scheduler
        params = model.parameters()

        self.total_step = len(self.train_loader)

        self.optimizer = torch.optim.Adam(params, args.lr)
        # self.optimizer = torch.optim.Adagrad(
        #     filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr
        # )

        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=len(self.train_loader) * args.epochs,
            eta_min=args.lr / 1000,
        )
        self.lr_scheduler.step()
        # Loss funcitons
        self.loss_fun = StructureLoss()

        # Metrics
        self.dice, self.iou = AvgMeter(), AvgMeter()
        self.best_iou = 0

        if args.mode == "test":
            weight = torch.load(f"{self.save_folder}/best.pth")
            self.model.load_state_dict(weight, strict=True)

    def train(self, epoch):
        self.model.train()

        tbar = tqdm(self.train_loader)
        losses = AvgMeter()

        for i, (data, mask) in enumerate(tbar):
            data = data.to(self.device)
            labels = mask.to(self.device)

            pred = self.model(data)
            loss = 0

            loss = self.loss_fun(pred, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            dice_score = dice_m(pred, mask)
            iou_score = iou_m(pred, mask)
            self.dice.update(dice_score.data, self.args.batchsize)
            self.iou.update(iou_score.data, self.args.batchsize)

            losses.update(loss.item(), pred.size(0))
            tbar.set_description("Epoch %d, loss %.4f" % (epoch, losses.avg))

        self.lr_scheduler.step()

    def test(self, epoch):
        self.model.eval()
        self.dice.reset()
        self.iou.reset()
        tbar = tqdm(self.val_loader)

        with torch.no_grad():
            for i, (data, mask) in enumerate(tbar):

                data = data.to(self.device)
                mask = mask.to(self.device)

                pred = self.model(data)

                dice_score = dice_m(pred, mask)
                iou_score = iou_m(pred, mask)
                self.dice.update(dice_score.data, self.args.batchsize)
                self.iou.update(iou_score.data, self.args.batchsize)

                tbar.set_description("Epoch %d" % (epoch))
            mean_IoU = self.iou.show()

            if self.mode == "train":
                if mean_IoU > self.best_iou:
                    self.best_iou = mean_IoU

                    torch.save(self.model.state_dict(), self.save_folder + "/best.pth")
                    # with open(osp.join(self.save_folder, "metric.log"), "a") as f:
                    #     f.write(
                    #         "{} - {:04d}\t - IoU {:.4f}\t - PD {:.4f}\t - FA {:.4f}\n".format(
                    #             time.strftime(
                    #                 "%Y-%m-%d-%H-%M-%S", time.localtime(time.time())
                    #             ),
                    #             epoch,
                    #             self.best_iou,
                    #             PD[0],
                    #             FA[0] * 1000000,
                    #         )
                    #     )

                all_states = {
                    "net": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "epoch": epoch,
                    "iou": self.best_iou,
                }
                torch.save(all_states, self.save_folder + "/checkpoint.pth")
            elif self.mode == "test":
                print("mIoU: " + str(mean_IoU) + "\n")


if __name__ == "__main__":
    args = parse_args()

    trainer = Trainer(args)
    if trainer.mode == "train":
        for epoch in range(trainer.start_epoch, args.epochs):
            trainer.train(epoch)
            trainer.test(epoch)
    else:
        trainer.test(1)
