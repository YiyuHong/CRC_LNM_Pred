#####################
# Train end to end
# freeze pretrained on imagenet model's layers
# add attention module at the end
# wsi batch n
# train all in slide patch
#####################
import sys
import os
import numpy as np
import argparse
import openslide
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import roc_curve, auc, f1_score
from efficientnet_pytorch import EfficientNet
from radam import RAdam, PlainRAdam
from Image_Augmentation import Image_Augmentation, Image_Augmentation_Alb
import os
import sys
import openslide
from PIL import Image
import numpy as np
import random
import argparse
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
import torchvision.models as models
from torch.nn.modules.normalization import LayerNorm
from torch.nn.init import xavier_uniform_
import math
from sklearn.metrics import cohen_kappa_score
import pandas as pd

parser = argparse.ArgumentParser(description='PANDA2020')
args = parser.parse_args()
args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

args.wsi_dir = r'/Arontier/People/hongyiyu/Data/Samsung_Colon2/Slides_200828'
args.val_lib = r'/Arontier/People/hongyiyu/Data/Samsung_Colon2/Tile_Dicts_200828/TS1024_OL0_Fold2_CV5/Test.pth'

args.num_class = 2
args.tile_size = 256
args.tile_batch_size = 10000


args.wsi_batch_size = 1
args.wsi_tile_max = 1500
args.wsi_tile_sampling_ratio_train = 1
args.wsi_tile_sampling_ratio_val = 1

args.workers = 16

args.model_name = 'resnet18'
args.attention_input_size = 512  # b0->1280 #b1->1280 b2->1408 b3->1536 b4->1792 b5->2048 r18,34->512 r50,101,152->2048

args.layer4_skip_flag = False
args.balance_flag = False
args.clinical_feature_num = 11
args.clinical_feature_flag = False
args.freeze_mode = 0  # -1->no freeze, 0-> last block, 1->last 2 blocks, 2->freeze all
args.lr = 0.001  # 0.001
args.dropout = 0

args.patch_probs_save_flag = True

args.output = r'/Arontier/People/hongyiyu/Data/Samsung_Colon2/Tile_Dicts_200828/TS1024_OL0_Fold2_CV5/Results/R18_P256_B64_Fold5/R18_FLB_P256_AO_DO25_B1_M1500_E8/Test_8'
if not os.path.exists(args.output):
    os.makedirs(args.output)
args.save_model_every = 1

# args.model_path = r''
args.model_path = r'/Arontier/People/hongyiyu/Data/Samsung_Colon2/Tile_Dicts_200828/TS1024_OL0_Fold2_CV5/Results/R18_P256_B64_Fold5/R18_FLB_P256_AO_DO25_B1_M1500_E8/checkpoint_8.pth'

# args.ft_model_path = r'E:\Projects\Colon\Data\Samsung_Colon\Slides_200828\Tile_Dicts\Sample\TS1024_OL0\Results\R34_P512_B4_B_W_Fold2\checkpoint_4.pth'


args.ft_model_path = r''

# fold1->9, fold2-> 16, fold3-> 13
# fold1->20, fold2-> 20, fold3-> 15


def main():
    # global args
    print(args)
    # score model
    resume_epoch = 0

    model = res_atten_net(ft_dim=args.attention_input_size, ft_model_path=args.ft_model_path,
                          num_class=args.num_class, freeze_mode=args.freeze_mode,
                          clinical_feature_flag=args.clinical_feature_flag, dropout=args.dropout)
    model.to(args.device)

    if args.model_path != '':
        print(f"load {args.model_path}")
        model_checkpoint = torch.load(args.model_path)
        model.load_state_dict(model_checkpoint['state_dict'])
        model.to(args.device)
        resume_epoch = model_checkpoint['epoch']

    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    attention_val_trans = transforms.Compose([transforms.Resize(args.tile_size), transforms.ToTensor(), normalize])

    val_dset = attention_data(args.val_lib, attention_transform=attention_val_trans)

    attention_val_loader = torch.utils.data.DataLoader(
        val_dset,
        batch_size=args.tile_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    # weights = [1, 2]
    # class_weights = torch.FloatTensor(weights).to(args.device)
    # criterion = nn.CrossEntropyLoss(weight=class_weights)
    criterion = nn.CrossEntropyLoss().to(args.device)
    # open output file
    fconv = open(os.path.join(args.output, 'result_test_log_' + str(resume_epoch) + '.csv'), 'a')
    fconv.write('epoch, val_loss, val_auc, tnr_tpr100\n')
    fconv.close()

    the_epoch_val_targets, the_epoch_val_slide_probs, the_epoch_val_slide_attens, val_loss = val_single(resume_epoch, model, val_dset,
                                                                            attention_val_loader, criterion)

    val_results, val_slide_probs_df = cal_patient_metrics(val_dset, the_epoch_val_slide_probs)

    val_slide_probs_df = pd.DataFrame(val_slide_probs_df)
    val_slide_probs_df.to_csv(os.path.join(args.output, f"slide_probs_test_{resume_epoch}.csv"), index=False)

    print(f"Val\tEpoch: {resume_epoch}\tVal_Loss: {val_loss}\t Val_AUC: {val_results['auc']}")

    fconv = open(os.path.join(args.output, 'result_test_log_' + str(resume_epoch) + '.csv'), 'a')
    fconv.write(
        '{}, {:.4f}, {:.4f}, {:.4f}\n'.format(resume_epoch, val_loss, val_results['auc'], val_results['tnr_tpr100']))
    fconv.close()

    if args.patch_probs_save_flag:
        patch_probs = {
                       'patch_attentions': the_epoch_val_slide_attens,
                       'slide_probs': np.float16(np.array(the_epoch_val_slide_probs)),
                       }
        torch.save(patch_probs, os.path.join(args.output, 'test_output_'+ str(resume_epoch) + '.pth'))

def cal_slide_metrics(slide_targets, slide_probs):
    slide_probs = np.array(slide_probs)
    slide_targets = np.array(slide_targets)

    fpr, tpr, thres = roc_curve(slide_targets, slide_probs)
    roc_auc = auc(fpr, tpr)
    f1 = f1_score(slide_targets, np.array(slide_probs) > 0.5)
    results = {
        'fpr': fpr,
        'tpr': tpr,
        'thres': thres,
        'auc': roc_auc,
        'f1': f1
    }
    return results


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


def val_single(epoch, model, val_dset, attention_train_loader, criterion):
    model.eval()
    val_dset.setmode(3)

    slide_num = len(val_dset.slidenames)
    slide_idxs = [i for i in range(slide_num)]
    slide_idxs_batch_sampled = [slide_idxs[i:i + 1] for i in
                                range(0, len(slide_idxs), 1)]

    running_loss = 0.
    the_epoch_val_targets = []
    the_epoch_val_slide_probs = []
    the_epoch_val_slide_attens = []
    for slide_batch_i, slide_idxs_batch in enumerate(slide_idxs_batch_sampled):

        batch_slide_targets = []
        batch_slide_clinical_features = []
        slide_idxs = slide_idxs_batch[0]
        the_slide_target = torch.from_numpy(np.array(val_dset.targets[slide_idxs])).to(args.device, dtype=torch.long)
        batch_slide_targets.append(the_slide_target)
        the_slide_clinical_features = torch.from_numpy(np.array(val_dset.clinical_features[slide_idxs])).to(args.device,
                                                                                                            dtype=torch.float)
        batch_slide_clinical_features.append(the_slide_clinical_features)
        the_epoch_val_targets.append(val_dset.targets[slide_idxs])

        batch_slide_targets = torch.stack(batch_slide_targets).to(args.device)
        batch_slide_clinical_features = torch.stack(batch_slide_clinical_features).to(args.device)

        the_slide_len = len(np.where(np.array(val_dset.slideIDX) == slide_idxs)[0])

        # print(slide_batch_patch_idxs)
        sampled_loss = []
        batch_slide_output_softmax_list = []

        if the_slide_len // args.wsi_tile_max == 0:
            val_slide_ensemble_num = 1
        else:
            val_slide_ensemble_num = the_slide_len // args.wsi_tile_max + 2

        for repeat_i in range(val_slide_ensemble_num):
            slide_batch_patch_idxs = val_dset.make_the_slide_tile_data_val(slide_idxs_batch)
            with torch.no_grad():
                for i, input in enumerate(attention_train_loader):
                    input = input.to(args.device)
                    batch_slide_output, A = model(input, batch_slide_clinical_features, slide_batch_patch_idxs)
                    the_slide_attention = print_Attention(A)
                    loss = criterion(batch_slide_output, batch_slide_targets)
                    sampled_loss.append(loss.item())
            batch_slide_output_softmax = F.softmax(batch_slide_output)
            batch_slide_output_softmax_list.extend(batch_slide_output_softmax.detach()[:, 1].cpu().tolist())
        running_loss += np.mean(sampled_loss)
        print(f"slide name: {val_dset.slidenames[slide_idxs]}")
        print(
            f'Val\tEpoch: [{epoch + 1}/{epoch + 1}]\tSlide Batch: [{slide_batch_i + 1}/{len(slide_idxs_batch_sampled)}]\tLoss: {np.mean(sampled_loss)}')
        the_epoch_val_slide_probs.append(np.mean(batch_slide_output_softmax_list))
        the_epoch_val_slide_attens.append(the_slide_attention)
    the_epoch_val_targets = np.array(the_epoch_val_targets)
    the_epoch_val_slide_probs = np.array(the_epoch_val_slide_probs).squeeze()
    the_epoch_val_slide_attens = np.array(the_epoch_val_slide_attens)
    return the_epoch_val_targets, the_epoch_val_slide_probs, the_epoch_val_slide_attens, running_loss / (slide_batch_i + 1)


def print_Attention(Total_A):
    for i, A in enumerate(Total_A):
        A_numpy = A.detach().cpu().numpy().squeeze()
        A_histo = np.histogram(A_numpy, bins=5)
        A_histo_bins = ['%.2f' % item for item in A_histo[1].tolist()]
        print(
            f'Slide {i} Attention -> Tile Num: {A_histo[0].sum()}\tMax Attention Value: {A_numpy.max():.5f}\t Histo: {A_histo[0].tolist()} {A_histo_bins}')

        return A_numpy.tolist()

def freeze_effi_net(model, mode=0):
    if mode == 0:
        for param in model._conv_stem.parameters():
            param.requires_grad = False
        for param in model._bn0.parameters():
            param.requires_grad = False
        for param in model._blocks[:-5].parameters():
            param.requires_grad = False
        print(f"blocks before last {-5} layers are frozen")

    elif mode == 1:
        for param in model._blocks[1::2].parameters():
            param.requires_grad = False
        print(f"odd layers are frozen")
    elif mode == 2:
        for param in model._conv_stem.parameters():
            param.requires_grad = False
        for param in model._bn0.parameters():
            param.requires_grad = False
        for param in model._blocks[::2].parameters():
            param.requires_grad = False
        print(f"even layers are frozen")
    else:
        print("Wrong Freeze Mode!!!!")
        exit(0)
    return model


def freeze_res_net(model, mode=0):
    if mode == 0:
        for param in model.conv1.parameters():
            param.requires_grad = False
        for param in model.bn1.parameters():
            param.requires_grad = False
        for param in model.layer1.parameters():
            param.requires_grad = False
        for param in model.layer2.parameters():
            param.requires_grad = False
        for param in model.layer3.parameters():
            param.requires_grad = False
        print(f"blocks before last layers are frozen")

    elif mode == 1:
        for param in model.conv1.parameters():
            param.requires_grad = False
        for param in model.bn1.parameters():
            param.requires_grad = False
        for param in model.layer1.parameters():
            param.requires_grad = False
        for param in model.layer2.parameters():
            param.requires_grad = False
        print(f"blocks before last 2 layers are frozen")
    elif mode == 2:
        for param in model.parameters():
            param.requires_grad = False
        print(f"feature extraction model is frozen")

    else:
        print("Wrong Freeze Mode!!!!")
        exit(0)
    return model


class Attn_Net_Gated(nn.Module):

    def __init__(self, L=1024, D=256, dropout=False, n_tasks=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_tasks)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A


class res_atten_net(nn.Module):
    def __init__(self, ft_dim=512, num_class=2, ft_model_path='', freeze_mode=-1, clinical_feature_flag=False,
                 dropout=0):
        super(res_atten_net, self).__init__()
        self.ft_dim = ft_dim
        self.num_class = num_class
        self.clinical_feature_flag = clinical_feature_flag
        # self.ft_model = EfficientNet.from_name(effi_name, override_params={'num_classes': self.num_class})
        self.ft_model = models.resnet18(pretrained=False, num_classes=num_class)

        if ft_model_path != '':
            ch = torch.load(ft_model_path)
            self.ft_model.load_state_dict(ch['state_dict'])

        if args.layer4_skip_flag:
            self.ft_model.layer4 = nn.Identity()

        self.ft_model.fc = nn.Identity()
        if freeze_mode > -1:
            self.ft_model = freeze_res_net(self.ft_model, freeze_mode)  ##

        self.attention_ori = nn.Sequential(
            nn.Linear(ft_dim, ft_dim // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(ft_dim // 2, 1)
        )

        # self.attention_gated = Attn_Net_Gated(L = ft_dim, D = ft_dim//2, dropout = True, n_tasks = 1)

        if self.clinical_feature_flag:
            ft_cf_dim = ft_dim + args.clinical_feature_num
        else:
            ft_cf_dim = ft_dim

        if args.wsi_batch_size > 1:
            self.classifier = nn.Sequential(
                nn.Linear(ft_cf_dim, ft_cf_dim // 2),
                nn.ReLU(),
                nn.BatchNorm1d(ft_cf_dim // 2),
                nn.Dropout(dropout),
                nn.Linear(ft_cf_dim // 2, self.num_class),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(ft_cf_dim, ft_cf_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(ft_cf_dim // 2, self.num_class),
            )

    def forward(self, x, slide_batch_clinical_features, slide_batch_patch_idxs):
        x = self.ft_model(x)
        x = x.view(x.size(0), -1)

        H = x.unsqueeze(1)
        slide_batch_M = []
        slide_batch_A = []
        for i in range(len(slide_batch_patch_idxs) - 1):
            the_slide_features = H[slide_batch_patch_idxs[i]:slide_batch_patch_idxs[i + 1], :, :]
            # print(the_slide_features.shape)

            the_slide_A = self.attention_ori(the_slide_features)  # NxK
            the_slide_A = the_slide_A.squeeze(2)

            # the_slide_A = self.attention_gated(the_slide_features)
            # the_slide_A = the_slide_A.squeeze(2)

            the_slide_A = torch.transpose(the_slide_A, 1, 0)  # KxN
            the_slide_A = F.softmax(the_slide_A, dim=1)  # softmax over N
            the_slide_M = torch.mm(the_slide_A, the_slide_features.squeeze(1))  # KxL
            if self.clinical_feature_flag:
                the_slide_M = torch.cat([the_slide_M, slide_batch_clinical_features[i].unsqueeze(dim=0)], dim=1)
            slide_batch_M.append(the_slide_M)
            slide_batch_A.append(the_slide_A)

        slide_batch_M = torch.stack(slide_batch_M).squeeze(dim=1).squeeze(dim=1).to(args.device)

        Y_prob = self.classifier(slide_batch_M)

        return Y_prob, slide_batch_A


class attention_data(data.Dataset):

    def __init__(self, path, attention_transform=None, inference_transform=None):

        lib = torch.load(path)
        self.attention_transform = attention_transform
        self.inference_transform = inference_transform
        self.mode = None
        self.targets = []
        self.slidenames = []
        self.grid = []
        self.slideIDX = []

        # remove empty slides
        empty_i_list = []
        for i, grids in enumerate(lib['grid']):
            if len(grids) == 0:
                print(lib['slides'][i])
                print('Empty Slide: {}'.format(lib['slides'][i]))
                empty_i_list.append(i)
        for index in sorted(empty_i_list, reverse=True):
            lib['targets'].pop(index)
            lib['slides'].pop(index)
            lib['grid'].pop(index)
            lib['patient_IDs'].pop(index)
            lib['clinical_features'].pop(index)

        self.patient_IDs = lib['patient_IDs']
        self.clinical_features = lib['clinical_features']

        self.targets = lib['targets']
        self.slidenames = lib['slides']

#         #         # for test
#         for i, g in enumerate(lib['grid']):
#             self.grid.extend(g[:30])
#             self.slideIDX.extend([i] * len(g[:30]))

        for i, g in enumerate(lib['grid']):
            self.grid.extend(g)
            self.slideIDX.extend([i] * len(g))

        self.level = lib['level']
        self.dict_tile_size = lib['dict_tile_size']
        self.dict_overlap = lib['dict_overlap']
        print('Dict Tile Size: {}'.format(self.dict_tile_size))
        print('Dict Overlap: {}'.format(self.dict_overlap))
        print('Number of tiles: {}'.format(len(self.grid)))

        slides = []
        for i, name in enumerate(lib['slides']):
            sys.stdout.write('Opening SVS headers: [{}/{}]\r'.format(i + 1, len(lib['slides'])))
            sys.stdout.flush()
            slide_path = os.path.join(args.wsi_dir, name)
            slide_path = slide_path.replace("\\", "/")

            slides.append(openslide.OpenSlide(slide_path))
        print('')
        self.slides = slides

    def setmode(self, mode):
        self.mode = mode

    def make_train_slide_data_balanced(self, wsi_batch_size):
        targets = np.array(self.targets)
        negative_slide_ID_idxs = np.where(targets == 0)[0]
        positive_slide_ID_idxs = np.where(targets == 1)[0]
        num_class_for_batch = wsi_batch_size // 2
        slide_idxs_batch_sampled = []
        for i in range(len(self.targets) // wsi_batch_size):
            selected_negative_slide_ID_idxs = random.sample(list(negative_slide_ID_idxs), num_class_for_batch)
            selected_positive_slide_ID_idxs = random.sample(list(positive_slide_ID_idxs), num_class_for_batch)
            slide_idxs_batch_sampled.append(selected_negative_slide_ID_idxs + selected_positive_slide_ID_idxs)

        return slide_idxs_batch_sampled

    def make_the_slide_tile_data_train(self, selected_slide_idxs):
        the_batch_slide_grid_idxs = []
        slide_batch_patch_idxs = [0]
        slideIDX = np.array(self.slideIDX)
        for selected_slide_idx in selected_slide_idxs:
            the_slide_grid_idxs = np.where(slideIDX == selected_slide_idx)[0]

            selected_tile_num_ratio = random.uniform(args.wsi_tile_sampling_ratio_train, 1)
            selected_tile_num = int(len(the_slide_grid_idxs) * selected_tile_num_ratio)

            if args.wsi_tile_max > selected_tile_num:
                the_batch_slide_grid_idxs.extend(the_slide_grid_idxs)
                slide_batch_patch_idxs.append(slide_batch_patch_idxs[-1] + selected_tile_num)

            else:
                the_batch_slide_grid_idxs.extend(
                    np.random.choice(the_slide_grid_idxs, args.wsi_tile_max, replace=False))
                # the_batch_slide_grid_idxs.extend(the_slide_grid_idxs[:args.wsi_tile_max])
                slide_batch_patch_idxs.append(slide_batch_patch_idxs[-1] + args.wsi_tile_max)

        self.t_data = [(self.slideIDX[x], self.grid[x]) for x in the_batch_slide_grid_idxs]

        return slide_batch_patch_idxs

    def make_the_slide_tile_data_val(self, selected_slide_idxs):
        the_batch_slide_grid_idxs = []
        slide_batch_patch_idxs = [0]
        slideIDX = np.array(self.slideIDX)
        for selected_slide_idx in selected_slide_idxs:
            the_slide_grid_idxs = np.where(slideIDX == selected_slide_idx)[0]

            selected_tile_num_ratio = random.uniform(args.wsi_tile_sampling_ratio_val, 1)
            selected_tile_num = int(len(the_slide_grid_idxs) * selected_tile_num_ratio)

            if args.wsi_tile_max > selected_tile_num:
                the_batch_slide_grid_idxs.extend(the_slide_grid_idxs)
                slide_batch_patch_idxs.append(slide_batch_patch_idxs[-1] + selected_tile_num)

            else:
                the_batch_slide_grid_idxs.extend(
                    np.random.choice(the_slide_grid_idxs, args.wsi_tile_max, replace=False))
                # the_batch_slide_grid_idxs.extend(the_slide_grid_idxs[:args.wsi_tile_max])
                slide_batch_patch_idxs.append(slide_batch_patch_idxs[-1] + args.wsi_tile_max)

        self.t_data = [(self.slideIDX[x], self.grid[x]) for x in the_batch_slide_grid_idxs]

        return slide_batch_patch_idxs

    def __getitem__(self, index):
        if self.mode == 1:
            out = []
            slideIDX, coord, target = self.t_data[index]

            for i in range(self.s):
                img = self.slides[slideIDX[i]].read_region(coord[i], self.level,
                                                           (self.dict_tile_size, self.dict_tile_size)).convert('RGB')

                if self.attention_transform is not None:
                    img = self.attention_transform(img)
                out.append(img)
            return out, target[0]

        elif self.mode == 2:
            slideIDX, coord = self.t_data[index]
            img = self.slides[slideIDX].read_region(coord, self.level,
                                                    (self.dict_tile_size, self.dict_tile_size)).convert('RGB')

            if self.inference_transform is not None:
                img = self.inference_transform(img)
            return img
        elif self.mode == 3:
            slideIDX, coord = self.t_data[index]
            img = self.slides[slideIDX].read_region(coord, self.level,
                                                    (self.dict_tile_size, self.dict_tile_size)).convert('RGB')

            if self.attention_transform is not None:
                img = self.attention_transform(img)
            return img

    def __len__(self):
        if self.mode == 1:
            return len(self.t_data)

        elif self.mode == 2:
            return len(self.t_data)
        elif self.mode == 3:
            return len(self.t_data)


if __name__ == '__main__':
    main()


