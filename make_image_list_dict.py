import sys
import os
import numpy as np
import argparse
import random
import openslide
import torch
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import KFold
import sys
import os
import numpy as np
import argparse
import random
import openslide
import PIL.Image as Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.metrics import roc_curve, auc
from efficientnet_pytorch import EfficientNet
from torch.utils.data.dataloader import DataLoader
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from PIL import Image, ImageDraw, ImageOps
import skimage


class Grid_Dataset(torch.utils.data.Dataset):
    def __init__(self, Grid_List, Slide_Path, Tile_Size, Downsample_Times=128):
        self.Grid_List = Grid_List
        self.Slide_Path = Slide_Path
        self.Tile_Size = Tile_Size
        self.Downsample_Times = Downsample_Times
    def __getitem__(self, index):
        Grid_Point = self.Grid_List[index]

        Slide_P = openslide.OpenSlide(self.Slide_Path)
        img = Slide_P.read_region(Grid_Point, 0, (self.Tile_Size, self.Tile_Size)).convert('RGB')
        img = img.resize((self.Tile_Size//self.Downsample_Times, self.Tile_Size//self.Downsample_Times))

        img_gray = Slide_P.read_region(Grid_Point, 0, (self.Tile_Size, self.Tile_Size)).convert('L')
        Slide_P.close()

        img_gray = np.array(img_gray)
        img_binary = img_gray > 220

        if np.mean(img_binary) > 0.7:
            Tissue_Flag = False
        elif np.sum(img_gray == 0) > self.Tile_Size * self.Tile_Size * 0.1:
            Tissue_Flag = False
        else:
            Tissue_Flag = True

        return np.array(Grid_Point), Tissue_Flag, np.array(img)

    def __len__(self):
        return len(self.Grid_List)


def Make_Grid_At_Level0(Slide_Path, Tile_Size, Overlap, Batch_Size=16, Num_Worker=4, Downsample_Times=128):
    Grid = []
    Slide_P = openslide.OpenSlide(Slide_Path)

    slide_width, slide_height = Slide_P.level_dimensions[0]

    for x in range(0, slide_width-Tile_Size, Tile_Size-Overlap):
        for y in range(0, slide_height-Tile_Size, Tile_Size-Overlap):
            Grid.append([x, y])

    ## Check image
    mask_width = slide_width//Downsample_Times+50 #downsample shape issue
    mask_height = slide_height//Downsample_Times+50
    Mask = Image.new('RGB', (mask_width, mask_height), 0)
    Mask = np.array(Mask)
    Mask_Dict = np.array(Mask)
    Mask_Ori = Mask_Dict.copy()

    resized_tile_size = Tile_Size//Downsample_Times
    Slide_P.close()

    No_BG_Grid = []
    Slide_Grid_Dataset = Grid_Dataset(Grid, Slide_Path, Tile_Size, Downsample_Times=Downsample_Times)
    Grid_Loader = DataLoader(Slide_Grid_Dataset, batch_size= Batch_Size, num_workers= Num_Worker, shuffle=False)
    for i, (Grid_Points, Tissue_Flags, Imgs) in enumerate(Grid_Loader):
        print('Get Total Grid Batch: [{}/{}]'.format(i + 1, len(Grid_Loader)))
        Tissue_Flags = Tissue_Flags.tolist()
        # if not any(Tissue_Flags):
        #     continue

        Tissue_Grid_Points = np.array(Grid_Points)[Tissue_Flags].tolist()
        No_BG_Grid.extend(Tissue_Grid_Points)
        # Imgs = np.array(Imgs)[Tissue_Flags,:,:,:]

        # for i_grid, dump_grid_point in enumerate(Tissue_Grid_Points):
        #     x, y = np.array(dump_grid_point)//Downsample_Times
        #     Mask_Dict[y:y+resized_tile_size, x:x+resized_tile_size, :] = Imgs[i_grid,:,:,:]

        for i_grid, dump_grid_point in enumerate(Grid_Points):
            x, y = np.array(dump_grid_point)//Downsample_Times
            Mask_Ori[y:y+resized_tile_size, x:x+resized_tile_size, :] = Imgs[i_grid,:,:,:]
            if Tissue_Flags[i_grid]:
                Mask_Dict[y:y + resized_tile_size, x:x + resized_tile_size, :] = Imgs[i_grid, :, :, :]

    return No_BG_Grid, Mask_Ori, Mask_Dict


if __name__ == '__main__':

    csv_col = ['LN_status', 'EMR_number', 'Subpath','Age','Sex', 'Comorbidity','FHX_CRC','Smoking','Alcohol','Shape_CFS', 'Location','Weight','Height','BMI', 'EMR_date',]
    clinical_feature_names = ['Age','Sex', 'Comorbidity','FHX_CRC','Smoking','Alcohol','Shape_CFS', 'Location','Weight','Height','BMI']

    Level = 0
    Tile_Size = 512
    Overlap = 0
    CV_Fold = 5

    Batch_Size = 4
    Num_Worker = 4

    Tile_Dict_Save_Dir = r'/Arontier/People/hongyiyu/Data/Samsung_Colon2/Tile_Dicts_200828/TS512_OL0'
    if not os.path.exists(Tile_Dict_Save_Dir):
        os.makedirs(Tile_Dict_Save_Dir)
    Check_Image_Save_Path = os.path.join(Tile_Dict_Save_Dir,'Check_0.7_220')
    if not os.path.exists(Check_Image_Save_Path):
        os.makedirs(Check_Image_Save_Path)


    WSI_Slides_Dir = r'/Arontier/People/hongyiyu/Data/Samsung_Colon2/Slides_200828'
    WSI_Level_Annot_List_Path = r'/Arontier/People/hongyiyu/Data/Samsung_Colon2/Slides_200828/WSI_Total_Slide.csv'
    WSI_Annot_Data = pd.read_csv(WSI_Level_Annot_List_Path) #image_id, isup_grade

    Total_Dict = {
        'patient_IDs': [],
        'slides': [],
        'grid': [],
        'targets': [],
        'clinical_features': [],
        'level': Level,
        'dict_tile_size': Tile_Size,
        'dict_overlap': Overlap,
    }

    Total_Dict['patient_IDs'] = WSI_Annot_Data['EMR_number'].values.tolist()
    Total_Dict['slides'] = WSI_Annot_Data['Subpath'].values.tolist()
    Total_Dict['targets'] = WSI_Annot_Data['LN_status'].values.tolist()
    Total_Dict['clinical_features'] = WSI_Annot_Data[clinical_feature_names].values.tolist()

    Empty_i_List = []

    for i, Slide_Name in enumerate(Total_Dict['slides']):
        # Slide_Path = r'E:\Projects\ECDP2020\Slides\TestDataset\Slides\73.mrxs'
        Slide_Path = os.path.join(WSI_Slides_Dir, Slide_Name)
        print(Slide_Path)


        Slide_Grid, Slide_Img_Downsampled, Dict_Downsampled_Img = Make_Grid_At_Level0(Slide_Path, Tile_Size, Overlap, Batch_Size=Batch_Size, Num_Worker=Num_Worker, Downsample_Times=16)
        print(f"length of the slide: {len(Slide_Grid)}")
        ### if empty slide
        if len(Slide_Grid)==0:
            Empty_i_List.append(i)
            Total_Dict['grid'].append([])
            continue
        # plt.figure()
        # plt.imshow(Check_Grid_Img)
        # plt.show()
        Total_Dict['grid'].append(Slide_Grid)


        Slide_Img_Downsampled_Path = os.path.join(Check_Image_Save_Path, os.path.split(Slide_Path)[1][:-4] + '_Ori.jpg')
        Dict_Downsampled_Img_Path = os.path.join(Check_Image_Save_Path, os.path.split(Slide_Path)[1][:-4] + '_Dict.jpg')

        plt.figure(figsize=(20, 40))
        plt.imshow(Slide_Img_Downsampled)
        plt.savefig(Slide_Img_Downsampled_Path)
        plt.close()

        plt.figure(figsize=(20, 40))
        plt.imshow(Dict_Downsampled_Img)
        plt.savefig(Dict_Downsampled_Img_Path)
        plt.close()


    ## remove empty slide
    print(f"Empty slide num: {len(Empty_i_List)} \n Empty slides name:")
    for index in sorted(Empty_i_List, reverse=True):
        print(Total_Dict['slides'][index])
        Total_Dict['targets'].pop(index)
        Total_Dict['slides'].pop(index)
        Total_Dict['grid'].pop(index)
        Total_Dict['clinical_features'].pop(index)

    MIL_Dict_Save_Path = os.path.join(Tile_Dict_Save_Dir, 'Total_Dict.pth')
    torch.save(Total_Dict, MIL_Dict_Save_Path)

    # split into cross-validation dataset
    Patient_IDs, patient_IDs_index = np.unique(WSI_Annot_Data['EMR_number'].values, return_index=True)
    patient_targets = WSI_Annot_Data['LN_status'].values[patient_IDs_index]
    skf = StratifiedKFold(n_splits=CV_Fold, random_state=0, shuffle=True)
    train_index_list = []
    val_index_list = []
    for i, (train_index, val_index) in enumerate(skf.split(Patient_IDs, patient_targets)):
        train_index_list.append(train_index.tolist())
        val_index_list.append(val_index.tolist())

    print(train_index_list)
    print(val_index_list)

    i_Fold = 0
    Processes = ['Train', 'Val']
    for train_index, val_index in zip(train_index_list, val_index_list):
        print('####################')
        print(train_index)
        print(val_index)
        i_Fold = i_Fold+1
        for Train_Val_Tag in Processes:
            if Train_Val_Tag == 'Train':
                indexes = np.array(train_index)
            elif Train_Val_Tag == 'Val':
                indexes = np.array(val_index)
            ## Dictionary
            MIL_Dict = {
                'patient_IDs': [],
                'slides': [],
                'grid': [],
                'targets': [],
                'clinical_features': [],
                'level': Level,
                'dict_tile_size': Tile_Size,
                'dict_overlap': Overlap,
            }
            Patient_Targets = []
            for index in indexes:
                Patient_ID = Patient_IDs[index]
                dump_index = np.array(WSI_Annot_Data.index[WSI_Annot_Data['EMR_number'] == Patient_ID])

                dump_slides = list(np.array(Total_Dict['slides'])[dump_index.astype(int)])
                dump_targets = list(np.array(Total_Dict['targets'])[dump_index.astype(int)])
                dump_targets = [int(i) for i in dump_targets]
                dump_grid = list(np.array(Total_Dict['grid'])[dump_index.astype(int)])
                dump_clinical_features = list(np.array(Total_Dict['clinical_features'])[dump_index.astype(int)])

                Patient_Targets.append(dump_targets[0])

                MIL_Dict['patient_IDs'].extend([Patient_ID]*len(dump_index))

                MIL_Dict['slides'].extend(dump_slides)
                MIL_Dict['targets'].extend(dump_targets)
                MIL_Dict['grid'].extend(dump_grid)
                MIL_Dict['clinical_features'].extend(dump_clinical_features)

            fconv = open(os.path.join(Tile_Dict_Save_Dir, 'CV_Statistics.csv'), 'a')
            fconv.write(f"{i_Fold},{Train_Val_Tag}, positive patients:,{Patient_Targets.count(1)}, negative patients:,{Patient_Targets.count(0)}\n")
            fconv.write(f"{i_Fold},{Train_Val_Tag}, positive slides:,{MIL_Dict['targets'].count(1)}, negative slides:,{MIL_Dict['targets'].count(0)}\n")
            fconv.close()

            MIL_Dict_Save_Path = os.path.join(Tile_Dict_Save_Dir,'CV_{}_{}.pth'.format(i_Fold, Train_Val_Tag))

            torch.save(MIL_Dict, MIL_Dict_Save_Path)
