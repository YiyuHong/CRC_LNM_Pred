import sys
import os
import numpy as np
import argparse
import random
import openslide
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import roc_curve, auc
from efficientnet_pytorch import EfficientNet
from radam import RAdam
from Image_Augmentation import Image_Augmentation, Image_Augmentation_Alb, Image_Augmentation_Alb_Strong
import time
from sklearn.metrics import roc_curve, auc, f1_score
import pandas as pd

parser = argparse.ArgumentParser(description='ECDP2020 tile classifier training script')
parser.add_argument('--train_lib', type=str, default='', help='path to train image tile list dictionary')
parser.add_argument('--val_lib', type=str, default='',
                    help='path to validation image tile list dictionary. If present.')
parser.add_argument('--output', type=str, default='', help='result output directory')
parser.add_argument('--batch_size', type=int, default=512, help='mini-batch size (default: 512)')
parser.add_argument('--nepochs', type=int, default=100, help='number of epochs')
parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--test_every', default=10, type=int, help='test on val every (default: 10)')
parser.add_argument('--model', type=str, default='', help='path to train resume model')

args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

args.wsi_dir = r'/Arontier/People/hongyiyu/Data/Samsung_Colon2/Slides_200828/Stain_V'
args.train_lib = r'/Arontier/People/hongyiyu/Data/Samsung_Colon2/Tile_Dicts_200828/TS512_OL0/CV_1_Train.pth'
args.val_lib = r'/Arontier/People/hongyiyu/Data/Samsung_Colon2/Tile_Dicts_200828/TS512_OL0/CV_1_Val.pth'

args.num_class = 2
args.tile_size = 256
args.batch_size = 64  # 24GB   600->5 , 456->12, 380->32, 300->64, 260, 240->128, 224->192
args.nepochs = 1000
args.workers = 8
args.test_every = 1
args.balance_flag = False
args.output = r'/Arontier/People/hongyiyu/Data/Samsung_Colon2/Tile_Dicts_200828/TS512_OL0/Results/SV_R34_P256_B64_Fold1'

if not os.path.exists(args.output):
    os.makedirs(args.output)

# args.model_path = r'/Arontier/People/hongyiyu/Data/Samsung_Colon2/Tile_Dicts_200828/TS256_OL0_Year_CV/Results/R34_P256_B64_2010/checkpoint_6.pth'
args.model_path = r''

args.model_name = 'resnet34'  # efficientnet-b0, resnet34


def main():
    global args, device
    print(args)

    resume_epoch = 0
    model = models.resnet34(pretrained=False, num_classes=args.num_class)

    #     model = models.resnet34(pretrained=False, num_classes=args.num_class)
    #     model = EfficientNet.from_name(args.model_name, override_params={'num_classes': args.num_class})

    model.to(device)
    optimizer = RAdam(model.parameters())

    if os.path.isfile(args.model_path):
        ch = torch.load(args.model_path)
        resume_epoch = ch['epoch']
        model.load_state_dict(ch['state_dict'])
        optimizer.load_state_dict(ch['optimizer'])
        model.to(device)

    # normalization
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    train_trans = transforms.Compose([Image_Augmentation_Alb(args.tile_size), transforms.ToTensor(), normalize])
#     train_trans = transforms.Compose([Image_Augmentation_Alb_Strong(args.tile_size), transforms.ToTensor(), normalize])

    val_trans = transforms.Compose([transforms.Resize(args.tile_size), transforms.ToTensor(), normalize])

    # load data
    train_dset = MILdataset(args.train_lib, train_trans)
    train_loader = DataLoader(
        train_dset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)
    if args.val_lib:
        val_dset = MILdataset(args.val_lib, val_trans)
        val_loader = DataLoader(
            val_dset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=False)

#     weights = [1, 2]
#     class_weights = torch.FloatTensor(weights).to(device)
#     criterion = nn.CrossEntropyLoss(weight=class_weights)
    criterion = nn.CrossEntropyLoss().to(device)

    fconv = open(os.path.join(args.output, 'result_log.csv'), 'a')
    fconv.write('epoch, train_loss, auc_patient_mean, tnr_at_tpr100_mean, auc_patient_minmax, tnr_at_tpr100_minmax\n')
    fconv.close()
    # loop throuh epochs
    for epoch in range(resume_epoch, args.nepochs):
        ## balance train data, for each mini-batch, first randomly select batch size number of WSIs in half positive and half negative,
        ## then, from each WSI randomly select 1 tile
        if args.balance_flag:
            train_dset.maketraindata_balanced(batch_size=args.batch_size, num_wsi_tile_for_batch=1)
        else:
            train_dset.maketraindata_ori()
            train_dset.shuffletraindata()

        train_dset.setmode(2)
        train_loss = train(epoch, train_loader, model, criterion, optimizer)

        print('Training\tEpoch: [{}/{}]\tLoss: {}'.format(epoch + 1, args.nepochs, train_loss))

        # Validation
        if args.val_lib and (epoch + 1) % args.test_every == 0:
            val_dset.setmode(1)
            probs = inference(epoch, val_loader, model)
            slide_mean_probs, slide_minmax_probs = group_val_slidenames(val_dset, probs)

            mean_results_patient, mean_slide_probs_df = cal_patient_metrics(val_dset, slide_mean_probs)
            mean_slide_probs_df = pd.DataFrame(mean_slide_probs_df)
            mean_slide_probs_df.to_csv(os.path.join(args.output, f"slide_probs_{epoch + 1}.csv"), index=False)

            minmax_results_patient, _ = cal_patient_metrics(val_dset, slide_minmax_probs)

            print('Validation\tEpoch: [{}/{}]\tAUC_Mean: {}\tTNR_Mean: {}\tAUC_Mean: {}\tTNR_MinMax: {}'
                  .format(epoch + 1, args.nepochs,
                          mean_results_patient['auc'],
                          mean_results_patient['tnr_tpr100'],
                          minmax_results_patient['auc'],
                          minmax_results_patient['tnr_tpr100'], ))
            fconv = open(os.path.join(args.output, 'result_log.csv'), 'a')
            fconv.write('{}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}\n'.format(epoch + 1, train_loss, mean_results_patient['auc'],
                                                          mean_results_patient['tnr_tpr100'],
                                                          minmax_results_patient['auc'],
                                                          minmax_results_patient['tnr_tpr100']))
            fconv.close()

            obj = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_loss': train_loss,
                'mean_results_patient': mean_results_patient,
                'minmax_results_patient': minmax_results_patient,
                'input_size': args.tile_size,
                'dict_tile_size': train_dset.dict_tile_size,
                'dict_overlap': train_dset.dict_overlap
            }
            torch.save(obj, os.path.join(args.output, f"checkpoint_{epoch + 1}.pth"))


def cal_patient_metrics(val_dset, slide_probs):
    slide_probs = np.array(slide_probs)
    slide_probs_df = {'slide_name': val_dset.slidenames, 'slide_probs': slide_probs,
                      'slide_targets': np.array(val_dset.targets)}
    patient_IDs_unique, patient_IDs_unique_index = np.unique(val_dset.patient_IDs, return_index=True)
    targets = np.array(val_dset.targets)
    patient_targets = targets[patient_IDs_unique_index]
    patient_probs = []

    for patient_ID in patient_IDs_unique:
        the_patient_probs = slide_probs[np.where(np.array(val_dset.patient_IDs) == patient_ID)[0]]
        patient_probs.append(np.mean(the_patient_probs))

    fpr, tpr, thres = roc_curve(patient_targets, patient_probs)
    tpr_100_index = np.where(tpr == 1)[0][0]
    tnr_at_tpr_100 = 1 - fpr[tpr_100_index]
    roc_auc = auc(fpr, tpr)
    f1 = f1_score(patient_targets, np.array(patient_probs) > 0.5)
    results = {
        'fpr': fpr,
        'tpr': tpr,
        'thres': thres,
        'auc': roc_auc,
        'tnr_tpr100': tnr_at_tpr_100,
        'f1': f1
    }

    return results, slide_probs_df


def inference(run, loader, model):
    model.eval()
    probs = torch.FloatTensor(len(loader.dataset))
    with torch.no_grad():
        i = 0
        for input in loader:
            print('Inference\tEpoch: [{}/{}]\tBatch: [{}/{}]'.format(run + 1, args.nepochs, i + 1, len(loader)))
            input = input.to(device)
            output = F.softmax(model(input), dim=1)
            probs[i * args.batch_size:i * args.batch_size + input.size(0)] = output.detach()[:, 1].clone()
            i = i + 1
    return probs.cpu().numpy()


def train(run, loader, model, criterion, optimizer):
    model.train()
    running_loss = 0.
    #### test balance mini-batch
    # for i, (input, target, slide_idx) in enumerate(loader):
    #     target = list(target.cpu().numpy())
    #     slide_idx = list(slide_idx.cpu().numpy())
    #     print('##############################################')
    #     print(f"0: {target.count(0)}, 1: {target.count(1)}")
    #     print(f"slide num: {len(np.unique(slide_idx))}")
    #     print(slide_idx)
    #     print('\n\n')
    ####
    for i, (input, target) in enumerate(loader):
        input = input.to(device)
        target = target.to(device)
        output = model(input)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * input.size(0)
        print('Train\tEpoch: [{}/{}]\tBatch: [{}/{}]\tLoss: {}'.format(run + 1, args.nepochs, i + 1, len(loader),
                                                                       loss.data.cpu().numpy()))

    return running_loss / len(loader.dataset)


def calc_err(pred, real):
    pred = np.array(pred)
    real = np.array(real)
    neq = np.not_equal(pred, real)
    err = float(neq.sum()) / pred.shape[0]
    fpr = float(np.logical_and(pred == 1, neq).sum()) / (real == 0).sum()
    fnr = float(np.logical_and(pred == 0, neq).sum()) / (real == 1).sum()
    return err, fpr, fnr


def group_val_slidenames(val_dset, val_probs):
    groups = np.array(val_dset.slideIDX)
    slide_mean_probs = []
    slide_minmax_probs = []
    for i_group in np.unique(groups):
        a_slide_probs = val_probs[groups == i_group]
        slide_mean_probs.append(np.mean(a_slide_probs))
        if len(a_slide_probs) < 20:
            slide_minmax_prob = np.mean(a_slide_probs)
        else:
            a_slide_probs_sorted = np.sort(a_slide_probs)[::-1]
            slide_minmax_prob = np.mean(a_slide_probs_sorted[:len(a_slide_probs_sorted) // 20]) + np.mean(
                a_slide_probs_sorted[-(len(a_slide_probs_sorted) // 20):])

        slide_minmax_probs.append(slide_minmax_prob)
    return slide_mean_probs, slide_minmax_probs


def cal_roc_auc(val_dset, slide_mean_probs, slide_minmax_probs):
    fpr_mean, tpr_mean, thres_mean = roc_curve(val_dset.targets, slide_mean_probs)
    roc_auc_mean = auc(fpr_mean, tpr_mean)
    mean_results = {
        'fpr': fpr_mean,
        'tpr': tpr_mean,
        'thres': thres_mean,
        'auc': roc_auc_mean
    }
    fpr_minmax, tpr_minmax, thres_minmax = roc_curve(val_dset.targets, slide_minmax_probs)
    roc_auc_minmax = auc(fpr_minmax, tpr_minmax)
    minmax_results = {
        'fpr': fpr_minmax,
        'tpr': tpr_minmax,
        'thres': thres_minmax,
        'auc': roc_auc_minmax
    }
    return mean_results, minmax_results


class MILdataset(data.Dataset):
    def __init__(self, libraryfile='', transform=None):
        lib = torch.load(libraryfile)
        slides = []
        for i, slide_subpath in enumerate(lib['slides']):
            sys.stdout.write('Opening SVS headers: [{}/{}]\r'.format(i + 1, len(lib['slides'])))
            sys.stdout.flush()
            slide_path = os.path.join(args.wsi_dir, slide_subpath)
            slide_path = slide_path.replace("\\", "/")
            slides.append(openslide.OpenSlide(slide_path))
        #             slides.append(slide_path)

        print('')
        # Flatten grid
        grid = []
        slideIDX = []

        for i, g in enumerate(lib['grid']):
            grid.extend(g)
            slideIDX.extend([i] * len(g))

#         for i, g in enumerate(lib['grid']):
#             grid.extend(g[:30])
#             slideIDX.extend([i] * len(g[:30]))

        print('Number of tiles: {}'.format(len(grid)))
        self.patient_IDs = lib['patient_IDs']
        self.slidenames = lib['slides']
        self.slides = slides
        self.targets = lib['targets']
        self.grid = grid
        self.slideIDX = slideIDX
        self.transform = transform
        self.mode = None

        self.level = lib['level']
        self.dict_tile_size = lib['dict_tile_size']
        self.dict_overlap = lib['dict_overlap']
        print('Dict Tile Size: {}'.format(self.dict_tile_size))
        print('Dict Overlap: {}'.format(self.dict_overlap))

    def setmode(self, mode):
        self.mode = mode

    def maketraindata(self, idxs):
        self.t_data = [(self.slideIDX[x], self.grid[x], self.targets[self.slideIDX[x]]) for x in idxs]

    def maketraindata_balanced(self, batch_size, num_wsi_tile_for_batch=1):
        targets = np.array(self.targets)
        slideIDX = np.array(self.slideIDX)
        negative_slide_ID_idxs = np.where(targets == 0)[0]
        positive_slide_ID_idxs = np.where(targets == 1)[0]
        balanced_idxs = []
        num_class_for_batch = batch_size // 2
        if num_class_for_batch % num_wsi_tile_for_batch != 0:
            print('num_class_for_batch%num_wsi_tile_for_batch !=0 ')
            exit()

        # 1/4 epoch save checkpoint
        #         for i in range(len(self.grid)//batch_size//4):
        for i in range(len(self.grid) // batch_size):
            print(f"Balanced Dataset Training Index Construction: {i}/{len(self.grid) // batch_size}")
            selected_negative_slide_ID_idxs = random.sample(list(negative_slide_ID_idxs),
                                                            num_class_for_batch // num_wsi_tile_for_batch)
            selected_positive_slide_ID_idxs = random.sample(list(positive_slide_ID_idxs),
                                                            num_class_for_batch // num_wsi_tile_for_batch)
            for nn, pp in zip(selected_negative_slide_ID_idxs * num_wsi_tile_for_batch,
                              selected_positive_slide_ID_idxs * num_wsi_tile_for_batch):
                negative_slide_grid_idxs = np.where(slideIDX == nn)[0]
                positive_slide_grid_idxs = np.where(slideIDX == pp)[0]
                balanced_idxs.extend(random.sample(list(negative_slide_grid_idxs), 1))
                balanced_idxs.extend(random.sample(list(positive_slide_grid_idxs), 1))

        self.t_data = [(self.slideIDX[x], self.grid[x], self.targets[self.slideIDX[x]]) for x in balanced_idxs]
        unique_dataset_ratio = len(np.unique(balanced_idxs)) / len(balanced_idxs)
        return unique_dataset_ratio

    def maketraindata_ori(self):
        self.t_data = [(self.slideIDX[x], self.grid[x], self.targets[self.slideIDX[x]]) for x in range(len(self.grid))]

    def shuffletraindata(self):
        self.t_data = random.sample(self.t_data, len(self.t_data))

    def __getitem__(self, index):
        if self.mode == 1:
            slideIDX = self.slideIDX[index]
            coord = self.grid[index]

            #             slide_path = self.slides[slideIDX]
            #             slide_p = openslide.OpenSlide(slide_path)
            #             img = slide_p.read_region(coord, self.level, (self.dict_tile_size, self.dict_tile_size)).convert('RGB')
            #             slide_p.close()

            slide_p = self.slides[slideIDX]
            img = slide_p.read_region(coord, self.level, (self.dict_tile_size, self.dict_tile_size)).convert('RGB')

            if self.transform is not None:
                img = self.transform(img)
            return img

        elif self.mode == 2:

            slideIDX, coord, target = self.t_data[index]

            #             slide_path = self.slides[slideIDX]
            #             slide_p = openslide.OpenSlide(slide_path)
            #             img = slide_p.read_region(coord, self.level, (self.dict_tile_size, self.dict_tile_size)).convert('RGB')
            #             slide_p.close()

            slide_p = self.slides[slideIDX]
            img = slide_p.read_region(coord, self.level, (self.dict_tile_size, self.dict_tile_size)).convert('RGB')

            if self.transform is not None:
                img = self.transform(img)
            return img, target

        elif self.mode == 3:  # for simple test
            slideIDX, coord, target = self.t_data[index]
            # img = self.slides[slideIDX].read_region(coord,self.level,(self.dict_tile_size,self.dict_tile_size)).convert('RGB')

            # if self.transform is not None:
            #     img = self.transform(img)
            return 0, target, slideIDX

    def __len__(self):
        if self.mode == 1:
            return len(self.grid)
        elif self.mode == 2:
            return len(self.t_data)
        elif self.mode == 3:
            return len(self.t_data)


if __name__ == '__main__':
    main()
