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
import matplotlib.pyplot as plt
import skimage
from mpl_toolkits.axes_grid1 import make_axes_locatable


parser = argparse.ArgumentParser(description='PANDA2020')
args = parser.parse_args()
args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

args.wsi_dir = r'/Arontier/People/hongyiyu/Data/Samsung_Colon2/Slides_200828'
args.val_lib = r'/Arontier/People/hongyiyu/Data/Samsung_Colon2/Tile_Dicts_200828/TS1024_OL0_Fold2_CV5/Test.pth'

args.test_patch_probs_pth = r'/Arontier/People/hongyiyu/Data/Samsung_Colon2/Tile_Dicts_200828/TS1024_OL0_Fold2_CV5/Results/R18_FLB_P256_AO_D25_B1_M1500_EBest_Ens_6_9_8_9_12/test_ens_output.pth'
args.atten_high_img_save_dir = r'/Arontier/People/hongyiyu/Data/Samsung_Colon2/Tile_Dicts_200828/TS1024_OL0_Fold2_CV5/Results/R18_FLB_P256_AO_D25_B1_M1500_EBest_Ens_6_9_8_9_12/atten_high'
if not os.path.exists(args.atten_high_img_save_dir):
    os.makedirs(args.atten_high_img_save_dir)
args.output = r'/Arontier/People/hongyiyu/Data/Samsung_Colon2/Tile_Dicts_200828/TS1024_OL0_Fold2_CV5/Results/R18_FLB_P256_AO_D25_B1_M1500_EBest_Ens_6_9_8_9_12/atten_heatmap'

if not os.path.exists(args.output):
    os.makedirs(args.output)

def main():
    # global args
    print(args)

    val_dset = attention_data(args.val_lib)

    test_patch_probs = torch.load(args.test_patch_probs_pth)
    ens_slide_probs = np.array(test_patch_probs['slide_probs'])
    ens_atten_probs = test_patch_probs['patch_attens']


    for slide_i in range(len(val_dset.slidenames)):
        print(val_dset.slidenames[slide_i])
        if len(val_dset.grid[slide_i])>1500:
            print("tile num large than 1500, skip")
            continue
        draw_attention_map(val_dset, ens_atten_probs, ens_slide_probs, slide_i)


def normalize_0_1(x):
    normalized_x = (x - np.min(x)) / (np.max(x) - np.min(x)+1e-8)
    return normalized_x

def draw_attention_map(dset, total_A, slide_probs, slide_i):
    A = total_A[slide_i]
    downsample_times = 16
    jet_cmap = plt.get_cmap('jet')

    ######
    Normalized_A = normalize_0_1(A)

    if len(A)<25:
        A_cutoff = np.min(A)
    else:
        A_cutoff = np.sort(A)[::-1][24]

    slide_path = os.path.join(args.wsi_dir, dset.slidenames[slide_i])
    slide_target = dset.targets[slide_i]
    slide_prob = slide_probs[slide_i]
    atten_high_slide_img_save_dir = os.path.join(args.atten_high_img_save_dir, str(slide_target)+'_'+ str(slide_prob)[:5]+'_'+os.path.split(dset.slidenames[slide_i])[-1])
    if not os.path.exists(atten_high_slide_img_save_dir):
        os.makedirs(atten_high_slide_img_save_dir)

    slide_p = openslide.OpenSlide(slide_path)
    width, height = slide_p.level_dimensions[0]

    slide_img_downsampled = np.ones([height//downsample_times+32, width//downsample_times+32,3])
    attention_map_downsampled = np.ones([height//downsample_times+32, width//downsample_times+32,4])

    slide_grid = dset.grid[slide_i]
    tile_size_downsampled = dset.dict_tile_size//downsample_times
    for i, grid in enumerate(slide_grid):

        patch_images = slide_p.read_region(grid, 0,(dset.dict_tile_size, dset.dict_tile_size)).convert('RGB')
        patch_images = np.array(patch_images)
        patch_image_downsampled = skimage.transform.resize(patch_images, (tile_size_downsampled,tile_size_downsampled))
        attention_value = A[i]
        normalized_attention_value = Normalized_A[i]
        if attention_value>=A_cutoff:

            atten_high_img = skimage.transform.resize(patch_images, (dset.dict_tile_size//2, dset.dict_tile_size//2))
            atten_high_img = Image.fromarray(np.uint8(atten_high_img*255))
            atten_high_img.save(os.path.join(atten_high_slide_img_save_dir, str(attention_value)[:8] + str(i) + '_' + '.png'))
            # ori_x, ori_y = slide_grid[i]
            # slide_p = openslide.OpenSlide(slide_path)
            # atten_high_img = slide_p.read_region((ori_x, ori_y), 0, (dset.dict_tile_size, dset.dict_tile_size)).convert('RGB').resize((dset.dict_tile_size//4, dset.dict_tile_size//4))
            # atten_high_img.save(os.path.join(atten_high_slide_img_save_dir,str(i)+'_'+str(attention_value)+'.jpg'))

        x, y = np.array(grid)//downsample_times
        slide_img_downsampled[y: y+tile_size_downsampled, x: x+tile_size_downsampled,:] = patch_image_downsampled
        attention_color = jet_cmap(normalized_attention_value, alpha=0.5)
        attention_map_downsampled[y: y+tile_size_downsampled, x: x+tile_size_downsampled,:] = attention_map_downsampled[y: y+tile_size_downsampled, x: x+tile_size_downsampled,:]*attention_color
    slide_p.close()
    # blended = Image.blend(im1, im2, alpha=0.5)
    # blended.save("blended.png")
    plt.figure()
    plt.imshow(slide_img_downsampled)
    plt.axis('off')
    plt.savefig(os.path.join(args.output, os.path.split(dset.slidenames[slide_i])[-1]+'.png'), dpi=300, pad_inches=0)

    plt.figure()
    plt.imshow(attention_map_downsampled)
    plt.axis('off')
    plt.savefig(os.path.join(args.output, os.path.split(dset.slidenames[slide_i])[-1]+'_atten.png'), dpi=300, pad_inches=0)

    plt.figure()
    ax = plt.gca()
    ax.imshow(slide_img_downsampled)
    plt_ax = ax.imshow(attention_map_downsampled, cmap='jet')
    plt.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cb = plt.colorbar(plt_ax, cax=cax)
    cb.ax.tick_params(labelsize=16)
    plt.savefig(os.path.join(args.output, os.path.split(dset.slidenames[slide_i])[-1]+'_overlap.png'), dpi=300, pad_inches=0)
    plt.show()

def print_Attention(Total_A):
    for i, A in enumerate(Total_A):
        A_numpy = A.detach().cpu().numpy().squeeze()
        A_histo = np.histogram(A_numpy, bins=5)
        A_histo_bins = ['%.2f' % item for item in A_histo[1].tolist()]
        print(f'Slide {i} Attention -> Tile Num: {A_histo[0].sum()}\tMax Attention Value: {A_numpy.max():.5f}\t Histo: {A_histo[0].tolist()} {A_histo_bins}')


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
#             self.grid.extend(g[:10])
#             self.slideIDX.extend([i] * len(g[:10]))

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
#             slides.append(slide_path)

        print('')
        self.slides = slides

    def setmode(self, mode):
        self.mode = mode

    def make_train_slide_data_balanced(self, wsi_batch_size):
        targets = np.array(self.targets)
        negative_slide_ID_idxs = np.where(targets == 0)[0]
        positive_slide_ID_idxs = np.where(targets == 1)[0]
        if wsi_batch_size<2:
            wsi_batch_size=2
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
            selected_tile_num = int(len(the_slide_grid_idxs)*selected_tile_num_ratio)

            if args.wsi_tile_max > selected_tile_num:
                the_batch_slide_grid_idxs.extend(the_slide_grid_idxs)
                slide_batch_patch_idxs.append(slide_batch_patch_idxs[-1] + selected_tile_num)

            else:
                the_batch_slide_grid_idxs.extend(np.random.choice(the_slide_grid_idxs, args.wsi_tile_max, replace=False))
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
            selected_tile_num = int(len(the_slide_grid_idxs)*selected_tile_num_ratio)

            if args.wsi_tile_max > selected_tile_num:
                the_batch_slide_grid_idxs.extend(the_slide_grid_idxs)
                slide_batch_patch_idxs.append(slide_batch_patch_idxs[-1] + selected_tile_num)

            else:
                the_batch_slide_grid_idxs.extend(np.random.choice(the_slide_grid_idxs, args.wsi_tile_max, replace=False))
                # the_batch_slide_grid_idxs.extend(the_slide_grid_idxs[:args.wsi_tile_max])
                slide_batch_patch_idxs.append(slide_batch_patch_idxs[-1] + args.wsi_tile_max)

        self.t_data = [(self.slideIDX[x], self.grid[x]) for x in the_batch_slide_grid_idxs]

        return slide_batch_patch_idxs



    def __getitem__(self, index):
        if self.mode == 1:
            out = []
            slideIDX, coord, target = self.t_data[index]

            for i in range(self.s):
                img = self.slides[slideIDX[i]].read_region(coord[i], self.level,(self.dict_tile_size, self.dict_tile_size)).convert('RGB')

                if self.attention_transform is not None:
                    img = self.attention_transform(img)
                out.append(img)
            return out, target[0]

        elif self.mode == 2:
            slideIDX, coord = self.t_data[index]
            img = self.slides[slideIDX].read_region(coord, self.level,(self.dict_tile_size, self.dict_tile_size)).convert('RGB')

            if self.inference_transform is not None:
                img = self.inference_transform(img)
            return img
        elif self.mode == 3:
            slideIDX, coord = self.t_data[index]
            
#             slide_path = self.slides[slideIDX]
#             slide_p = openslide.OpenSlide(slide_path)
#             img = slide_p.read_region(coord, self.level, (self.dict_tile_size, self.dict_tile_size)).convert('RGB')
#             slide_p.close()
            
            img = self.slides[slideIDX].read_region(coord, self.level,(self.dict_tile_size, self.dict_tile_size)).convert('RGB')

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
