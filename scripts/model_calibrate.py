#!/usr/bin/env python
# coding: utf-8

import os,sys
sys.path.insert(0,"..")
from glob import glob
from os.path import exists, join
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import argparse

import torch
import torchvision, torchvision.transforms
import skimage.transform
import sklearn, sklearn.model_selection

import pickle
import random
import train_utils

import torchxrayvision as xrv


parser = argparse.ArgumentParser()
parser.add_argument('-f', type=str, default="", help='')
parser.add_argument('dataset', type=str, default="mimic_ch")
parser.add_argument('weights_filename', type=str,default = '')
parser.add_argument('-seed', type=int, default=0, help='')
parser.add_argument('-cuda', type=bool, default=True, help='')
parser.add_argument('-batch_size', type=int, default=8, help='')
parser.add_argument('-threads', type=int, default=4, help='')
parser.add_argument('-mdtable', action='store_true', help='')
parser.add_argument('--dataset_dir', type=str, default="")

cfg = parser.parse_args()

data_aug = None

transforms = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),xrv.datasets.XRayResizer(512)])

datas = []
datas_names = []
cfg.dataset = 'mimic_ch'
if "nih" in cfg.dataset:
    dataset = xrv.datasets.NIH_Dataset(
        imgpath=cfg.dataset_dir + "/images-512-NIH", 
        transform=transforms, data_aug=data_aug, unique_patients=False, views=["PA","AP"])
    datas.append(dataset)
    datas_names.append("nih")
if "pc" in cfg.dataset:
    dataset = xrv.datasets.PC_Dataset(
        imgpath=cfg.dataset_dir + "/images-512-PC", 
        transform=transforms, data_aug=data_aug, unique_patients=False, views=["PA","AP"])
    datas.append(dataset)
    datas_names.append("pc")
if "chex" in cfg.dataset:
    dataset = xrv.datasets.CheX_Dataset(
        imgpath=cfg.dataset_dir + "/CheXpert-v1.0-small",
        csvpath=cfg.dataset_dir + "/CheXpert-v1.0-small/train.csv",
        transform=transforms, data_aug=data_aug, unique_patients=False)
    datas.append(dataset)
    datas_names.append("chex")
if "google" in cfg.dataset:
    dataset = xrv.datasets.NIH_Google_Dataset(
        imgpath=cfg.dataset_dir + "/images-512-NIH",
        transform=transforms, data_aug=data_aug)
    datas.append(dataset)
    datas_names.append("google")
if "mimic_ch" in cfg.dataset:
    print('innnnnnnnnnnn')
    dataset = xrv.datasets.MIMIC_Dataset(
        imgpath='/share/nvmedata/santosh/Oxford/using_pretrained_v2_replacing_LA_v2/test_results/test_output_epoch=42_1_of_2_AP_test_temporal_two_studies_v5_0_5_cxrbert/GT',#"/share/ssddata/physionet.org/files/mimic-cxr-jpg/2.0.0/files/",
        csvpath=cfg.dataset_dir + "mimic-cxr-2.0.0-chexpert.csv.gz",
        metacsvpath=cfg.dataset_dir + "mimic-cxr-2.0.0-metadata.csv.gz",
        splitpath = '/share/nvmedata/santosh/Oxford/using_pretrained_v2_replacing_LA_v2/test_results/generation_classification_v1.csv', #cfg.dataset_dir + "mimic-cxr-2.0.0-split.csv.gz",
        transform=transforms, data_aug=data_aug, unique_patients=False, views=["PA","AP"])
    datas.append(dataset)
    datas_names.append("mimic_ch")
if "openi" in cfg.dataset:
    dataset = xrv.datasets.Openi_Dataset(
        imgpath=cfg.dataset_dir + "/OpenI/images/",
        transform=transforms, data_aug=data_aug)
    datas.append(dataset)
    datas_names.append("openi")
if "rsna" in cfg.dataset:
    dataset = xrv.datasets.RSNA_Pneumonia_Dataset(
        imgpath=cfg.dataset_dir + "/kaggle-pneumonia-jpg/stage_2_train_images_jpg",
        transform=transforms, data_aug=data_aug, unique_patients=False, views=["PA","AP"])
    datas.append(dataset)
    datas_names.append("rsna")
if "siim" in cfg.dataset:
    dataset = xrv.datasets.SIIM_Pneumothorax_Dataset(
        imgpath=cfg.dataset_dir + "/SIIM_TRAIN_TEST/dicom-images-train",
        csvpath=cfg.dataset_dir + "/SIIM_TRAIN_TEST/train-rle.csv",
        transform=transforms, data_aug=data_aug)
    datas.append(dataset)
    datas_names.append("siim")
if "vin" in cfg.dataset:
    dataset = xrv.datasets.VinBrain_Dataset(
        imgpath=cfg.dataset_dir + "vinbigdata-chest-xray-abnormalities-detection/train",
        csvpath=cfg.dataset_dir + "vinbigdata-chest-xray-abnormalities-detection/train.csv",
        transform=transforms, data_aug=data_aug)
    datas.append(dataset)
    datas_names.append("vin")
    
orj_dataset_pathologies = datas[0].pathologies
print('orj_dataset_pathologies', orj_dataset_pathologies)
# load model

if cfg.weights_filename == "jfhealthcare":
    model = xrv.baseline_models.jfhealthcare.DenseNet() 
elif cfg.weights_filename == "chexpert":
    model = xrv.baseline_models.chexpert.DenseNet(weights_zip="/home/users/joecohen/scratch/chexpert/chexpert_weights.zip")
else:
    print('elseeeeeeeeeee')
    model = xrv.models.get_model(cfg.weights_filename, apply_sigmoid=True)
    model.op_threshs = None

print("datas_names", datas_names)

for d in datas:
    print(d)
    xrv.datasets.relabel_dataset(model.pathologies, d)

#cut out training sets
train_datas = []
test_datas = []
for i, dataset in enumerate(datas):
    # give patientid if not exist
    if "patientid" not in dataset.csv:
        dataset.csv["patientid"] = ["{}-{}".format(dataset.__class__.__name__, i) for i in range(len(dataset))]
        
    # gss = sklearn.model_selection.GroupShuffleSplit(train_size=0.0001,test_size=0.9999, random_state=cfg.seed)
    
    # train_inds, test_inds = next(gss.split(X=range(len(dataset)), groups=dataset.csv.patientid))
    # print(train_inds)
    # train_dataset = xrv.datasets.SubsetDataset(dataset, train_inds)
    # test_dataset = xrv.datasets.SubsetDataset(dataset, test_inds)
    
    # train_datas.append(train_dataset)
    # test_datas.append(test_dataset)
    test_dataset = dataset
    
    test_datas.append(test_dataset)
    
if len(datas) == 0:
    raise Exception("no dataset")
elif len(datas) == 1:
#     train_dataset = train_datas[0]
    test_dataset = test_datas[0]
else:
    print("merge datasets")
    train_dataset = xrv.datasets.Merge_Dataset(train_datas)
    test_dataset = xrv.datasets.Merge_Dataset(test_datas)

# Setting the seed
np.random.seed(cfg.seed)
random.seed(cfg.seed)
torch.manual_seed(cfg.seed)
if cfg.cuda:
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# print("train_dataset.labels.shape", train_dataset.labels.shape)
print("test_dataset.labels.shape", test_dataset.labels.shape)
# print("train_dataset",train_dataset)
print("test_dataset",test_dataset)
#print(model)

test_loader = torch.utils.data.DataLoader(test_dataset,
                                           batch_size=cfg.batch_size,
                                           shuffle=False,
                                           num_workers=cfg.threads, pin_memory=cfg.cuda)

filename = "1_2_GT_with_thresh_LA" + os.path.basename(cfg.weights_filename).split(".")[0] + "_" + "-".join(datas_names) + ".pkl"
print(filename)
filename = os.path.join('/home/santoshsanjeev/Oxford/torchxrayvision/scripts/final_results_santosh', filename)
if os.path.exists(filename):
    print("Results already computed")
    results = pickle.load(open(filename, "br"))
else:
    print("Results are being computed")
    if cfg.cuda:
        model = model.cuda()
    results = train_utils.valid_test_epoch("test", 0, model, "cuda", test_loader, torch.nn.BCEWithLogitsLoss(), limit=99999999)
    pickle.dump(results, open(filename, "bw"))
# print(results)
print("Model pathologies:",model.pathologies)
print("Dataset pathologies:",test_dataset.pathologies)

# perf_dict = {}
# all_threshs = []
# all_min = []
# all_max = []
# all_ppv80 = []
# for i, patho in enumerate(test_dataset.pathologies):
#     opt_thres = np.nan
#     opt_min = np.nan
#     opt_max = np.nan
#     ppv80_thres = np.nan
#     if (len(results[3][i]) > 0) and (len(np.unique(results[3][i])) == 2):
        
#         #sigmoid
#         all_outputs = 1.0/(1.0 + np.exp(-results[2][i]))
        
#         fpr, tpr, thres = sklearn.metrics.roc_curve(results[3][i], all_outputs)
#         pente = tpr - fpr
#         opt_thres = thres[np.argmax(pente)]
#         opt_min = all_outputs.min()
#         opt_max = all_outputs.max()
        
#         ppv, recall, thres = sklearn.metrics.precision_recall_curve(results[3][i], all_outputs)
#         ppv80_thres_idx = np.where(ppv > 0.8)[0][0]
#         ppv80_thres = thres[ppv80_thres_idx-1]
        
#         auc = sklearn.metrics.roc_auc_score(results[3][i], all_outputs)
        
#         print(patho, auc)
#         perf_dict[patho] = str(round(auc,2))
        
#     else:
#         perf_dict[patho] = "-"
        
#     all_threshs.append(opt_thres)
#     all_min.append(opt_min)
#     all_max.append(opt_max)
#     all_ppv80.append(ppv80_thres)
perf_dict = {}
all_threshs = []#[0.58074194, 0.54576606, np.nan, 0.5225241, 0.66747427, np.nan, np.nan, 0.7036588, 0.521654, np.nan, 0.5775543, np.nan, np.nan, np.nan, 0.57897043, 0.520638, 0.6373577, 0.5326836]
all_min = []
all_max = []
all_ppv80 = []
all_accuracy = []
all_f1_score = []
all_precision = []
all_recall = []
all_auc = []

for i, patho in enumerate(test_dataset.pathologies):
    print(i, patho)
    opt_thres = np.nan
    opt_min = np.nan
    opt_max = np.nan
    ppv80_thres = np.nan
    accuracy = np.nan
    f1_score = np.nan
    precision = np.nan
    recall = np.nan
    auc = np.nan
    
    if (len(results[3][i]) > 0) and (len(np.unique(results[3][i])) == 2):
        
        #sigmoid
        all_outputs = 1.0/(1.0 + np.exp(-results[2][i]))
        
        fpr, tpr, thres_roc = sklearn.metrics.roc_curve(results[3][i], all_outputs)
        pente = tpr - fpr
        opt_thres = thres_roc[np.argmax(pente)]
        opt_min = all_outputs.min()
        opt_max = all_outputs.max()
        
        ppv, recall, thres_pr = sklearn.metrics.precision_recall_curve(results[3][i], all_outputs)
        ppv80_thres_idx = np.where(ppv > 0.8)[0][0]
        ppv80_thres = thres_pr[ppv80_thres_idx-1]
        
        auc = sklearn.metrics.roc_auc_score(results[3][i], all_outputs)
        
        # Calculate confusion matrix for accuracy, precision, recall, and F1 score
        threshold = opt_thres #all_threshs[i]  
        predicted_labels = (all_outputs >= threshold).astype(int)
        true_labels = results[3][i]
        confusion_matrix = sklearn.metrics.confusion_matrix(true_labels, predicted_labels)
        TP = confusion_matrix[1, 1]
        TN = confusion_matrix[0, 0]
        FP = confusion_matrix[0, 1]
        FN = confusion_matrix[1, 0]

        # Calculate metrics
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1_score = 2 * (precision * recall) / (precision + recall)
        
        # Add metrics to perf_dict
        perf_dict[patho] = {
            'AUC': round(auc, 2),
            'Accuracy': round(accuracy, 2),
            'F1 Score': round(f1_score, 2),
            'Precision': round(precision, 2),
            'Recall': round(recall, 2)
        }
        
        all_auc.append(auc)  # Append AUC to the list
        
    else:
        perf_dict[patho] = "-"
    
    # Append metrics to respective lists
    all_threshs.append(opt_thres)
    all_min.append(opt_min)
    all_max.append(opt_max)
    all_ppv80.append(ppv80_thres)
    all_accuracy.append(accuracy)
    all_f1_score.append(f1_score)
    all_precision.append(precision)
    all_recall.append(recall)

# Print the results
print("pathologies", test_dataset.pathologies)
print("------------------------------------------------------------------------------------------------")
print("op_threshs", str(all_threshs).replace("nan", "np.nan"))
print("min", str(all_min).replace("nan", "np.nan"))
print("max", str(all_max).replace("nan", "np.nan"))
print("ppv80", str(all_ppv80).replace("nan", "np.nan"))
print("accuracy", str(all_accuracy).replace("nan", "np.nan"))
print("f1_score", str(all_f1_score).replace("nan", "np.nan"))
print("precision", str(all_precision).replace("nan", "np.nan"))
print("recall", str(all_recall).replace("nan", "np.nan"))
print("all AUC values:", str(all_auc).replace("nan", "np.nan"))

# Calculate and print average metrics
avg_accuracy = np.nanmean(all_accuracy)
avg_f1_score = np.nanmean(all_f1_score)
avg_precision = np.nanmean(all_precision)
avg_recall = np.nanmean(all_recall)
avg_auc = np.nanmean(all_auc)

print(f'Average Accuracy: {round(avg_accuracy, 2)}')
print(f'Average F1 Score: {round(avg_f1_score, 2)}')
print(f'Average Precision: {round(avg_precision, 2)}')
print(f'Average Recall: {round(avg_recall, 2)}')
print(f'Average AUC: {round(avg_auc, 2)}')
    
# print("pathologies",test_dataset.pathologies)
    
# print("op_threshs",str(all_threshs).replace("nan","np.nan"))
    
# print("min",str(all_min).replace("nan","np.nan"))
    
# print("max",str(all_max).replace("nan","np.nan"))

# print("ppv80",str(all_ppv80).replace("nan","np.nan"))
    
    
if cfg.mdtable:
    print("|Model Name|" + "|".join(orj_dataset_pathologies) + "|")
    print("|---|" + "|".join(("-"*len(orj_dataset_pathologies))) + "|")
    
    accs = [perf_dict[patho] if (patho in perf_dict) else "-" for patho in orj_dataset_pathologies]
    print("|"+str(model)+"|" + "|".join(accs) + "|")
    
print("Done")



