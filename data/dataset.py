from __future__ import print_function

from matplotlib import pyplot as plt
from torch.utils.data import DataLoader,Dataset
import numpy as np
import os
from PIL import Image
from skimage import io,transform
import cv2
import torch
import platform
import pandas as pd
import argparse, time, random
import yaml
from yaml.loader import SafeLoader
from tqdm import tqdm
import h5py
import gc
import math
import scipy.interpolate
from PIL import Image
import cv2
from matplotlib import pyplot as plt
from torchvision.transforms import Compose
from sklearn import preprocessing
# import transform.transforms_group as our_transform

class IvYGAP_Dataset(Dataset):
    def __init__(self, phase, args, if_end2end=False):
        super(IvYGAP_Dataset, self).__init__()
        self.args = args
        self.patc_bs=64
        self.phase=phase
        self.if_end2end=if_end2end
        
        labels_path = self.args.dataDir+'IvYGAP/multimodal_diag_survival_IvY.csv'
        
        excel_label_wsi = pd.read_csv(labels_path, header=0)
        excel_wsi = excel_label_wsi.values
        
        PATIENT_LIST = excel_wsi[:,0]
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)
        PATIENT_LIST = list(PATIENT_LIST)

        PATIENT_LIST = np.unique(PATIENT_LIST) 
        np.random.shuffle(PATIENT_LIST)
        NUM_PATIENT_ALL = len(PATIENT_LIST) 
        
        if args.novalset:
            TRAIN_PATIENT_LIST = PATIENT_LIST[0:int(NUM_PATIENT_ALL * 0.67)]
            TEST_PATIENT_LIST = PATIENT_LIST[int(NUM_PATIENT_ALL * 0.67):]
            
            self.TRAIN_LIST = []
            self.TEST_LIST = []
            for i in range(excel_wsi.shape[0]):
                if excel_wsi[:,0][i] in TRAIN_PATIENT_LIST:
                    self.TRAIN_LIST.append(excel_wsi[i,:])
                elif excel_wsi[:,0][i] in TEST_PATIENT_LIST:
                    self.TEST_LIST.append(excel_wsi[i,:])
            self.LIST= np.asarray(self.TRAIN_LIST) if self.phase == 'Train' else (np.asarray(self.VAL_LIST) if self.phase == 'Val' else np.asarray(self.TEST_LIST))

        else:
            TRAIN_PATIENT_LIST = PATIENT_LIST[0:int(NUM_PATIENT_ALL * 0.8)]
            VAL_PATIENT_LIST = PATIENT_LIST[int(NUM_PATIENT_ALL * 0.9):]
            TEST_PATIENT_LIST = PATIENT_LIST[int(NUM_PATIENT_ALL * 0.8):int(NUM_PATIENT_ALL * 0.9)]
            
            self.TRAIN_LIST = []
            self.VAL_LIST = []
            self.TEST_LIST = []
            for i in range(excel_wsi.shape[0]):
                if excel_wsi[:,0][i] in TRAIN_PATIENT_LIST:
                    self.TRAIN_LIST.append(excel_wsi[i,:])
                elif excel_wsi[:,0][i] in TEST_PATIENT_LIST:
                    self.TEST_LIST.append(excel_wsi[i,:])
                elif excel_wsi[:,0][i] in VAL_PATIENT_LIST:
                    self.VAL_LIST.append(excel_wsi[i,:])
            self.LIST= np.asarray(self.TRAIN_LIST) if self.phase == 'Train' else (np.asarray(self.VAL_LIST) if self.phase == 'Val' else np.asarray(self.TEST_LIST))

        self.train_iter_count=0
        self.Flat=0
        self.WSI_all=[]
        
        # for gene expression files:
        self.rows_genes = pd.read_csv(self.args.dataDir+'IvYGAP/gene_expression_matrix_2014-11-25/rows-genes.csv', header=0)
        self.columns_samples = pd.read_csv(self.args.dataDir+'IvYGAP/gene_expression_matrix_2014-11-25/columns-samples.csv', header=0)
        self.fpkm_table = pd.read_csv(self.args.dataDir+'IvYGAP/gene_expression_matrix_2014-11-25/fpkm_table.csv', header=0)
        
        # self.gene_list = pd.read_csv(self.args.dataDir+'IvYGAP/share_gene_signature.csv', header=0)
        share_gene_path = self.args.dataDir + 'TCGA' + '/gene_signature_selected.xlsx'
        self.share_gene = pd.read_excel(share_gene_path, sheet_name='0.3_high_exp', header=0)
        self.share_gene_tumor = self.share_gene[self.share_gene['Type'] == 'Tumor']
        self.share_gene_immune = self.share_gene[self.share_gene['Type'] == 'Immune']
        
        self.gene_selected = self.rows_genes[self.rows_genes['gene_symbol'].isin(self.share_gene['gene_symbol'].values.tolist())]
        self.gene_selected_tumor= self.rows_genes[self.rows_genes['gene_symbol'].isin(self.share_gene_tumor['gene_symbol'].values.tolist())]
        self.gene_selected_immune= self.rows_genes[self.rows_genes['gene_symbol'].isin(self.share_gene_immune['gene_symbol'].values.tolist())]
        
    
        self.fpkm_table = self.fpkm_table[self.fpkm_table['gene_id\\rna_well_id'].isin(self.gene_selected['gene_id'].values.tolist())]
        self.fpkm_table_tumor = self.fpkm_table[self.fpkm_table['gene_id\\rna_well_id'].isin(self.gene_selected_tumor['gene_id'].values.tolist())]
        self.fpkm_table_immune = self.fpkm_table[self.fpkm_table['gene_id\\rna_well_id'].isin(self.gene_selected_immune['gene_id'].values.tolist())]
        
        self.columns_samples['specimen_name_temp'] = self.columns_samples['specimen_name'].apply(lambda x: '-'.join(x.split('-')[:3]))
        self.columns_samples['patient_id_temp'] = self.columns_samples['specimen_name'].apply(lambda x: x.split('-')[0])
        
        # this is the shared survival interval between TCGA and IvYGAP, 
        # according to the patient-level survival time
        # for TCGA+IvYGAP all patients: 233.5 511.0 929.0

        # for TCGA+IvYGAP all_uncensored patients: 212.5 454.0 776.5
        if args.survival_interval == "uncensored":
            self.quantile_25 = 212.5
            self.quantile_50 = 454.0
            self.quantile_75 = 776.5
        else:
            self.quantile_25 = 233.5
            self.quantile_50 = 511.0
            self.quantile_75 = 929.0       

    def __getitem__(self, index):
        if not self.if_end2end:
            wsi_features = self.read_feature(index)
        else:
            wsi_features = self.read_img(index)
            
        gene_features, gene_features_tumor, gene_features_immune = self.read_gene(index)
        
        list_labels=self.label_generation(index)
        
        gene = torch.from_numpy(gene_features).float()
        gene_tumor = torch.from_numpy(gene_features_tumor).float()
        gene_immune = torch.from_numpy(gene_features_immune).float()
        
        return torch.from_numpy(np.array(wsi_features)).float(), gene, gene_tumor, gene_immune, torch.from_numpy(list_labels)

    def read_feature(self, index):
        root = self.args.dataDir + 'IvYGAP' + '/Res50_feature_'+str(self.args.fixdim)+'_fixdim0_norm/'
        patch_all = h5py.File(root + self.LIST[index, 1] + '.h5')['Res_feature'][:]
        return patch_all[0]

    def read_img(self, index):
        wsi_path = self.dataDir + self.LIST[index, 1]
        patch_all = []
        patch_all_ori = []
        coor_all = []
        coor_all_ori = []
        self.img_dir = os.listdir(wsi_path)

        read_details = np.load(self.args.dataDir + 'IvYGAP' + '/read_details/' + self.LIST[index, 1] + '.npy', allow_pickle=True)[0]
        num_patches = read_details.shape[0]
        max_num = self.args.fixdim
        Use_patch_num = num_patches if num_patches <= max_num else max_num
        if num_patches <= max_num:
            times = int(np.floor(max_num / num_patches))
            remaining = max_num % num_patches
            for i in range(Use_patch_num):
                img_temp = io.imread(wsi_path + '/' + str(read_details[i][0]) + '_' + str(read_details[i][1]) + '.jpg')
                patch_all_ori.append(img_temp)
                coor_all_ori.append(read_details[i])
            patch_all = patch_all_ori
            coor_all = coor_all_ori
            ####### fixdim0
            if times > 1:
                for k in range(times - 1):
                    patch_all = patch_all + patch_all_ori
                    coor_all = coor_all + coor_all_ori
            if not remaining == 0:
                patch_all = patch_all + patch_all_ori[0:remaining]

        else:
            for i in range(Use_patch_num):
                img_temp = io.imread(wsi_path + '/' + str(read_details[int(np.around(i * (num_patches / max_num)))][0]) + '_' + str(read_details[int(np.around(i * (num_patches / max_num)))][1]) + '.jpg')
                patch_all.append(img_temp)

        patch_all = np.asarray(patch_all)

        # data augmentation
        patch_all = patch_all.reshape(-1, 224, 3)  # (num_patches*28,28,3)
        patch_all = patch_all.reshape(-1, 224, 224, 3)  # (num_patches,28,28,3)
        patch_all = patch_all.reshape(max_num, -1)  # (num_patches,28*28*3)

        patch_all = patch_all / 255.0
        patch_all = patch_all.astype(np.float32)

        return patch_all

    def read_gene(self, index):
        '''
        gene_features.shape: (431,)
        '''
        wsi_id = self.LIST[index, 1]
        # print('wsi_id:', wsi_id) parts1[:3] == parts2[:3] W19-1-1-D.01
        wsi_id_parts = '-'.join(wsi_id.split('-')[:3])
        patient_id = wsi_id.split('-')[0]
        if not wsi_id_parts in self.columns_samples['specimen_name_temp'].values:
            if not patient_id in self.columns_samples['patient_id_temp'].values:
                print(patient_id, "not found") #W18, W31, W35
        assert wsi_id_parts in self.columns_samples['specimen_name_temp'].values, '{:}'.format(self.columns_samples['specimen_name_temp'])
        rna_well_id = self.columns_samples[self.columns_samples['specimen_name_temp'] == wsi_id_parts]['rna_well_id'].values[0]
        # print('1.rna_well_id:', rna_well_id)
        rna_well_id = str(rna_well_id)
        gene_features = self.fpkm_table[rna_well_id].values
        gene_features_tumor = self.fpkm_table_tumor[rna_well_id].values
        gene_features_immune = self.fpkm_table_immune[rna_well_id].values
        # print('2.IvYGAP gene_features:', gene_features)
        self.input_size_omic = gene_features.shape[0] #431
        self.input_size_omic_tumor = gene_features_tumor.shape[0] #59
        self.input_size_omic_immune = gene_features_immune.shape[0] #361
        
        return gene_features, gene_features_tumor, gene_features_immune
    

    def label_generation(self,index):
        
        # grade
        if self.LIST[index, 3]=='G2':
            label_Grade=0 #Grade 2
        elif self.LIST[index, 3] == 'G3':
            label_Grade = 1 #Grade 3
        else:
            label_Grade=2 #Grade 4
        
        # diag 2021
        if self.LIST[index, 4]=='WT':
            label_Diag = 0 #Grade4 GBM
        elif self.LIST[index, 5] == 'codel':
            label_Diag = 3 #Grade 2/3 Oligo
        else:
            if self.LIST[index, 6] == -2 or self.LIST[index, 6] == -1 or self.LIST[index, 3] =='G4':
                label_Diag = 1 #Grade 4 Astro
            else:
                label_Diag = 2 #Grade 2/3 Astro
                
        # subtyping
        if self.LIST[index, 4]=='WT':
            label_Subtype = 0 #Grade 4 GBM
        elif self.LIST[index, 5] == 'codel':
            label_Subtype = 2 #Grade 2/3 Oligo
        else:
            label_Subtype = 1 #Grade 2/3/4 Astro

        # survival_interval
        survival_time = self.LIST[index, -1]
        if survival_time < self.quantile_25:
            label_survival = 0 #Survival interval 0
        elif self.quantile_25 <= survival_time and survival_time < self.quantile_50:
            label_survival = 1 #Survival interval 1
        elif self.quantile_50 <= survival_time and survival_time < self.quantile_75:
            label_survival = 2 #Survival interval 2
        else:
            label_survival = 3 #Survival interval 3

        # survival times
        label_survival_time = survival_time
            
        # survival_censor
        if self.LIST[index, -2] == 1 :
            label_censor = 0 #if dead, Survival censor 0
            label_censored = 1
            label_event = 1
        else:
            label_censor = 1 #if live, Survival censor 1
            label_censored = 0
            label_event = 0

        # label=np.asarray([label_IDH,label_1p19q,label_CDKN,label_His,label_Grade,label_Diag,label_His_2class])
        temp = np.zeros_like(label_Diag)
        list_labels=np.asarray([temp,temp,temp,temp,label_Grade, label_Diag, temp, label_Subtype, label_survival, label_censor, label_event, label_survival_time])

        return  list_labels


    def shuffle_list(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        np.random.shuffle(self.LIST)


    def __len__(self):
        return self.LIST.shape[0]


class TCGA_Dataset(Dataset):
    def __init__(self, phase, args, if_end2end=False):
        super(TCGA_Dataset, self).__init__()
        self.args = args
        self.patc_bs=64
        self.phase=phase
        self.if_end2end=if_end2end
        
        labels_path = self.args.dataDir+'TCGA/multimodal_diag_survival_TCGA.csv'
        excel_label_wsi = pd.read_csv(labels_path, header=0)
        excel_wsi = excel_label_wsi.values
        
        PATIENT_LIST = excel_wsi[:,0]
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)
        PATIENT_LIST = list(PATIENT_LIST)

        PATIENT_LIST = np.unique(PATIENT_LIST)
        np.random.shuffle(PATIENT_LIST)
        NUM_PATIENT_ALL = len(PATIENT_LIST)
        
        if args.novalset:
            TRAIN_PATIENT_LIST = PATIENT_LIST[0:int(NUM_PATIENT_ALL * 0.67)]
            TEST_PATIENT_LIST = PATIENT_LIST[int(NUM_PATIENT_ALL * 0.67):]
            
            self.TRAIN_LIST = []
            self.TEST_LIST = []
            for i in range(excel_wsi.shape[0]):
                if excel_wsi[:,0][i] in TRAIN_PATIENT_LIST:
                    self.TRAIN_LIST.append(excel_wsi[i,:])
                elif excel_wsi[:,0][i] in TEST_PATIENT_LIST:
                    self.TEST_LIST.append(excel_wsi[i,:])
            self.LIST= np.asarray(self.TRAIN_LIST) if self.phase == 'Train' else (np.asarray(self.VAL_LIST) if self.phase == 'Val' else np.asarray(self.TEST_LIST))
        else:
            TRAIN_PATIENT_LIST = PATIENT_LIST[0:int(NUM_PATIENT_ALL * 0.8)]
            VAL_PATIENT_LIST = PATIENT_LIST[int(NUM_PATIENT_ALL * 0.9):]
            TEST_PATIENT_LIST = PATIENT_LIST[int(NUM_PATIENT_ALL * 0.8):int(NUM_PATIENT_ALL * 0.9)]
            
            self.TRAIN_LIST = []
            self.VAL_LIST = []
            self.TEST_LIST = []
            for i in range(excel_wsi.shape[0]):
                if excel_wsi[:,0][i] in TRAIN_PATIENT_LIST:
                    self.TRAIN_LIST.append(excel_wsi[i,:])
                elif excel_wsi[:,0][i] in VAL_PATIENT_LIST:
                    self.VAL_LIST.append(excel_wsi[i,:])
                elif excel_wsi[:,0][i] in TEST_PATIENT_LIST:
                    self.TEST_LIST.append(excel_wsi[i,:])
            self.LIST= np.asarray(self.TRAIN_LIST) if self.phase == 'Train' else (np.asarray(self.VAL_LIST) if self.phase == 'Val' else np.asarray(self.TEST_LIST))

        self.train_iter_count=0
        self.Flat=0
        self.WSI_all=[]
        
        share_gene_path = self.args.dataDir + 'TCGA' + '/gene_signature_selected.xlsx'
        self.share_gene = pd.read_excel(share_gene_path, sheet_name='0.3_high_exp', header=0)
        self.share_gene_tumor = self.share_gene[self.share_gene['Type'] == 'Tumor']
        self.share_gene_immune = self.share_gene[self.share_gene['Type'] == 'Immune']
        
        # this is the shared survival interval between TCGA and IvYGAP
        if args.survival_interval == "uncensored":
            self.quantile_25 = 212.5
            self.quantile_50 = 454.0
            self.quantile_75 = 776.5
        else:
            self.quantile_25 = 233.5
            self.quantile_50 = 511.0
            self.quantile_75 = 929.0   

    def __getitem__(self, index):
        if not self.if_end2end:
            wsi_features = self.read_feature(index)
        else:
            wsi_features = self.read_img(index)
            
        gene_features, gene_features_tumor, gene_features_immune = self.read_gene(index)

        list_labels=self.label_generation(index)
        
        gene = torch.from_numpy(gene_features).float()
        gene_tumor = torch.from_numpy(gene_features_tumor).float()
        gene_immune = torch.from_numpy(gene_features_immune).float()

        return torch.from_numpy(np.array(wsi_features)).float(), gene, gene_tumor, gene_immune, torch.from_numpy(list_labels)

    def read_feature(self, index):
        root = self.args.dataDir + 'TCGA' + '/Res50_feature_'+str(self.args.fixdim)+'_fixdim0_norm/'
        patch_all = h5py.File(root + self.LIST[index, 1] + '.h5')['Res_feature'][:]
        return patch_all[0]

    def read_img(self, index):
        wsi_path = self.dataDir + self.LIST[index, 1]
        patch_all = []
        patch_all_ori = []
        coor_all = []
        coor_all_ori = []
        self.img_dir = os.listdir(wsi_path)

        read_details = np.load(self.args.dataDir + 'TCGA' + '/read_details/' + self.LIST[index, 1] + '.npy', allow_pickle=True)[0]
        num_patches = read_details.shape[0]
        max_num = self.args.fixdim
        Use_patch_num = num_patches if num_patches <= max_num else max_num
        if num_patches <= max_num:
            times = int(np.floor(max_num / num_patches))
            remaining = max_num % num_patches
            for i in range(Use_patch_num):
                img_temp = io.imread(wsi_path + '/' + str(read_details[i][0]) + '_' + str(read_details[i][1]) + '.jpg')
                patch_all_ori.append(img_temp)
                coor_all_ori.append(read_details[i])
            patch_all = patch_all_ori
            coor_all = coor_all_ori
            ####### fixdim0
            if times > 1:
                for k in range(times - 1):
                    patch_all = patch_all + patch_all_ori
                    coor_all = coor_all + coor_all_ori
            if not remaining == 0:
                patch_all = patch_all + patch_all_ori[0:remaining]

        else:
            for i in range(Use_patch_num):
                img_temp = io.imread(wsi_path + '/' + str(read_details[int(np.around(i * (num_patches / max_num)))][0]) + '_' + str(read_details[int(np.around(i * (num_patches / max_num)))][1]) + '.jpg')
                patch_all.append(img_temp)

        patch_all = np.asarray(patch_all)

        # data augmentation
        patch_all = patch_all.reshape(-1, 224, 3)  # (num_patches*28,28,3)
        patch_all = patch_all.reshape(-1, 224, 224, 3)  # (num_patches,28,28,3)
        patch_all = patch_all.reshape(max_num, -1)  # (num_patches,28*28*3)

        patch_all = patch_all / 255.0
        patch_all = patch_all.astype(np.float32)

        return patch_all

    def read_gene(self, index):
        '''
        gene_features.shape: (431,)
        '''
        gene_path = self.args.dataDir + 'TCGA' + '/transcriptomeProfiling_geneExpression/' + self.LIST[index, 11] + '/' + self.LIST[index, 12]
        gene_df = pd.read_table(gene_path, skiprows=1)
        # remove the rebundant gene_name
        gene_df = gene_df.drop_duplicates(subset=['gene_name'], keep='first')
        gene_selected = gene_df[gene_df['gene_name'].isin(self.share_gene['gene_symbol'].values.tolist())]
        gene_features = gene_selected['fpkm_uq_unstranded'].values
        
        gene_selected_tumor = gene_df[gene_df['gene_name'].isin(self.share_gene_tumor['gene_symbol'].values.tolist())]
        gene_features_tumor = gene_selected_tumor['fpkm_uq_unstranded'].values     
        
        gene_selected_immune = gene_df[gene_df['gene_name'].isin(self.share_gene_immune['gene_symbol'].values.tolist())]
        gene_features_immune= gene_selected_immune['fpkm_uq_unstranded'].values
        
        self.input_size_omic = gene_features.shape[0] #431
        self.input_size_omic_tumor = gene_features_tumor.shape[0] #59
        self.input_size_omic_immune = gene_features_immune.shape[0] #361
        
        return gene_features, gene_features_tumor, gene_features_immune

    def label_generation(self,index):


        if self.LIST[index, 4]=='WT':
            label_IDH=0
        elif self.LIST[index, 4]=='Mutant':
            label_IDH=1
        if self.LIST[index, 5] == 'non-codel':
            label_1p19q = 0
        elif self.LIST[index, 5] == 'codel':
            label_1p19q = 1
        if self.LIST[index, 6] == -2 or self.LIST[index, 6] == -1:
            label_CDKN = 1
        else:
            label_CDKN = 0

        if self.LIST[index, 2]=='oligoastrocytoma':
            label_His = 0
        elif self.LIST[index, 2] == 'astrocytoma':
            label_His = 1
        elif self.LIST[index, 2] == 'oligodendroglioma':
            label_His = 2
        elif self.LIST[index, 2] == 'glioblastoma':
            label_His = 3

        if self.LIST[index, 2]=='glioblastoma':
            label_His_2class = 1
        else:
            label_His_2class = 0

        # grade
        if self.LIST[index, 3]=='G2':
            label_Grade=0
        elif self.LIST[index, 3] == 'G3':
            label_Grade = 1
        else:
            label_Grade=2 

        # Diag 2021
        if self.LIST[index, 4]=='WT':
            label_Diag = 0 #Grade4 GBM
        elif self.LIST[index, 5] == 'codel':
            label_Diag = 3 #Grade 2/3 Oligo
        else:
            if self.LIST[index, 6] == -2 or self.LIST[index, 6] == -1 or self.LIST[index, 3] =='G4':
                label_Diag = 1 #Grade 4 Astro
            else:
                label_Diag = 2 #Grade 2/3 Astro
                
        # subtyping
        if self.LIST[index, 4]=='WT':
            label_Subtype = 0 #Grade 4 GBM
        elif self.LIST[index, 5] == 'codel':
            label_Subtype = 2 #Grade 2/3 Oligo
        else:
            label_Subtype = 1 #Grade 2/3/4 Astro
                
        # survival_interval
        survival_time = self.LIST[index, -1]
        if survival_time < self.quantile_25:
            label_survival = 0 #Survival interval 0
        elif self.quantile_25 <= survival_time and survival_time < self.quantile_50:
            label_survival = 1 #Survival interval 1
        elif self.quantile_50 <= survival_time and survival_time < self.quantile_75:
            label_survival = 2 #Survival interval 2
        else:
            label_survival = 3 #Survival interval 3

        # survival times
        label_survival_time = survival_time

        # survival_censor
        if self.LIST[index, -2] == 1 :
            label_censor = 0 #if dead, Survival censor 0
            label_event = 1
        else:
            label_censor = 1 #if live, Survival censor 1
            label_event = 0

        
        list_labels=np.asarray([label_IDH,label_1p19q,label_CDKN,label_His,label_Grade,label_Diag,label_His_2class, label_Subtype, label_survival, label_censor, label_event, label_survival_time])

        return  list_labels


    def shuffle_list(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        np.random.shuffle(self.LIST)



    def __len__(self):
        return self.LIST.shape[0]
