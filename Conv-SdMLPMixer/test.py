import os
import json
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
import pandas as pd
from model import create_RepVGG_B1g2
#from fasternet import FasterNet
# from repvgg import create_RepVGG_B2,create_RepVGG_B1g2
# from model4 import resnet50
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import time
from sklearn.metrics import f1_score,precision_score,recall_score,roc_auc_score,accuracy_score
from sklearn.preprocessing import label_binarize
from torch.nn.parallel import DataParallel

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])   
    data_transform = {
        "val": transforms.Compose([
                                   transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   normalize
                                   ])}


    val_data = '/home/rzt/project_TransUNet/data/chest_xray/test'  


    validate_dataset = datasets.ImageFolder(val_data,
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  shuffle=True,
                                                  num_workers=16)

    print("{} images for test.".format(val_num))

    print('Using {} dataloader workers every process'.format(16))
    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = validate_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
 

    net = create_RepVGG_B1g2(num_classes=2).cuda()
    # net = nn.DataParallel(net)
    # load pretrain weights

    # model_weight_path = "./save_model/chest/faster06-07/val06-01.pth"
    model_weight_path = "./save_model/chest/RepVGGB2210-25/val10-25.pth"

    net.load_state_dict(torch.load(model_weight_path))
    
    net.to(device)


    val_steps = len(validate_loader)


    acc_val = []
  
    net.eval()
    auc = 0.0
    acccc = 0.0
    acc = 0.0  # accumulate accurate number / epoch
    precision=0.0
    recall=0.0
    F1_score=0.0
    predlist=[]
    scorelist=[]
    targetlist=[]

    labels = [0,1]
    with torch.no_grad():
        val_bar = tqdm(validate_loader)
        for val_data in val_bar:
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))
            val_lab = label_binarize(val_labels.cpu(), classes=labels)
            predict_y = torch.max(outputs, dim=1)[1]
            val_pre = label_binarize(predict_y.cpu(), classes=labels)
            # print(val_labels,predict_y.cpu())
            # exit()
            predlist=np.append(predlist, predict_y.cpu().numpy())
            targetlist=np.append(targetlist,val_labels)
        test_folder = './model_test/chest/ours(1){}'.format(time.strftime('%m-%d', time.localtime()))
        if not os.path.exists(test_folder):
            os.mkdir(test_folder)
        with open(test_folder+'/targetlist{}.txt'.format(time.strftime('%m-%d %H:%M', time.localtime())),'a+') as f:        
            for tar in targetlist:
                f.write(str(tar)+'\n')
                f.close
        with open(test_folder+'/predlist{}.txt'.format(time.strftime('%m-%d %H:%M', time.localtime())),'a+') as f:        
            for pre in predlist:
                f.write(str(pre)+'\n')
                f.close 
        acc = accuracy_score(targetlist, predlist, normalize=True)
        F1 = f1_score(targetlist, predlist)
        precision = precision_score(targetlist, predlist)
        recall = recall_score(targetlist, predlist)
        val_lab = label_binarize(targetlist, classes=labels)
        val_pre = label_binarize(predlist, classes=labels)
        auc = roc_auc_score(val_lab, val_pre)
        print('acc',acc)
        print('F1',F1)
        print('precision',precision)
        print('recall',recall)
        print('auc',auc)
        with open(test_folder+'/metrics{}.txt'.format(time.strftime('%m-%d %H:%M', time.localtime())),'a+') as f:  
            f.write('acc:{}, F1:{}, precision:{}, recall:{}, auc:{}\n'.format(acc,F1,precision,recall,auc))

    print('Finished Test！')


if __name__ == '__main__':
    main()
