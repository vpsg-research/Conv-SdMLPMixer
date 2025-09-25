import os
import json
from datetime import datetime
import torch
import time
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
import pandas as pd
#from repvgg import create_RepVGG_B2
from model import create_RepVGG_B1g2
# from resnet import *
# from model.effv2 import efficientnetv2_s
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import label_binarize
from PIL import ImageFile
from sklearn.metrics import f1_score,precision_score,recall_score,roc_auc_score,accuracy_score
from ptflops import get_model_complexity_info
from torch.nn.parallel import DataParallel

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# 定义画图函数
def matplot_loss(train_loss, val_loss,save_path):
    plt.plot(train_loss, label='train_loss')
    plt.plot(val_loss, label='val_loss')
    plt.legend(loc='best')
    plt.ylabel('loss-acc')
    plt.xlabel('epoch')
    plt.title("Accuracy-Loss")

    plt.savefig(save_path+'/'+'Loss.pdf',dpi=500,format='pdf')
    plt.savefig(save_path+'/'+'Loss.svg',dpi=500,bbox_inches = 'tight')
    # plt.show()
def matplot_acc(train_acc, val_acc,save_path):
    plt.plot(train_acc, label='train_acc')
    plt.plot(val_acc, label='val_acc')
    plt.legend(loc='best')
    plt.ylabel('loss-acc')
    plt.xlabel('epoch')
    plt.title("Accuracy-Loss")

    plt.savefig(save_path+'/'+'Accuracy.pdf',dpi=500,format='pdf')
    plt.savefig(save_path+'/'+'Accuracy.svg',dpi=500,bbox_inches = 'tight')
    # plt.show()
    
def main():

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     normalize,
                                     ]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   normalize
                                   ])}


    train_data = '/home/rzt/project_TransUNet/data/chest_xray/train'
    val_data = '/home/rzt/project_TransUNet/data/chest_xray/test'

    train_dataset = datasets.ImageFolder(train_data,
                                         transform=data_transform["train"])    

    train_num = len(train_dataset)

    classes_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in classes_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indicesskin.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 16])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(val_data,
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, 
                                                  shuffle=True,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    
    net = create_RepVGG_B1g2(num_classes=2)

    # net = nn.DataParallel(net)
    
    net.to(device)

       # 加载预训练权重 
    ###################################################begin
    net_dict = net.state_dict()
    pretrained_dict = torch.load("/home/rzt/project_TransUNet/fenlei/ours/save_model/chest/RepVGGB210-25/val_auc10-25.pth")
    for k, v in pretrained_dict.items():
        print(k)
        name = k.replace('module.','',1)
        print(name)
        net_dict[name] = v 
    net.load_state_dict(net_dict)
    # exit()
    ###################################################end
    
    # pre_weights=torch.load("./repvgg/save_model/isic05-18/val_repvgg05-18.pth",map_location=device)
    # pre_dict = {k: v for k, v in pre_weights.items() if net.state_dict()[k].numel() == v.numel()}
    # missing_keys, unexpected_keys = net.load_state_dict(pre_dict, strict=False)
    
    # change fc layer structure
    # net.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()
    # loss_function = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([0.44,0.73,0.83])).float(),reduction='mean')
    ### 要想使用GPU，此处必须使用cuda()
    # loss_function.to(device)
    
    
    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    epochs = 300
    train_best_acc = 0.0
    val_best_acc = 0.0
    val_best_re = 0.0
    val_best_f1 = 0.0
    val_best_pre = 0.0
    val_best_auc = 0.0


    train_steps = len(train_loader)
    val_steps = len(validate_loader)
    loss_train = []
    acc_train = []
    loss_val = []
    acc_val = []
    for epoch in range(epochs):
        # train
        net.train()
        acc0 = 0.0
        running_loss = 0.0
        predlist=[]
        scorelist=[]
        targetlist=[]
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predict_y0 = torch.max(logits, dim=1)[1]
            acc0 += torch.eq(predict_y0, labels.to(device)).sum().item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,epochs,loss)
       
        train_loss = running_loss / train_steps
        train_accurate = acc0 / train_num      
                    
        # validate
        net.eval()

        acc = 0.0  # accumulate accurate number / epoch
        valdata_loss = 0.0
        time1 = "%s"%datetime.now()#获取当前时间
        # labels = [0,1,2]#用于计算auc
        labels = [0,1]#用于计算auc
        with torch.no_grad():
            val_bar = tqdm(validate_loader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                
                outputs = net(val_images.to(device))
                # print(outputs)
   
                predict_y = torch.max(outputs, dim=1)[1]
                # print(predict_y,val_labels)
                # exit()
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                valloss = loss_function(outputs, val_labels.to(device))
                valdata_loss += valloss.item()
                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,epochs)
                predlist=np.append(predlist, predict_y.cpu().numpy())
                targetlist=np.append(targetlist,val_labels)
        val_loss = valdata_loss / val_steps
        val_accurate = acc / val_num
        
        #用于绘图
        loss_train.append(train_loss)
        acc_train.append(train_accurate)
        loss_val.append(val_loss)
        acc_val.append(val_accurate)
        # train_val_folder = './repvggB2g4/model_train_val/isic{}'.format(time.strftime('%m-%d', time.localtime()))
        train_val_folder = './model_train_val/chest/RepVGGB2(1){}'.format(time.strftime('%m-%d', time.localtime()))
        if not os.path.exists(train_val_folder):
            os.mkdir(train_val_folder) 
        list = [time1,epoch,train_loss,train_accurate,val_loss,val_accurate]
        #list以二维列表的形式转化为DataFrame
        data = pd.DataFrame([list])
        data.to_csv(train_val_folder+'/'+'train_log.csv',mode='w',header=False,index=False)#mode设为a,就可以向csv文件追加数据了
        output = "%s：Step [%d],  train_loss : %f, train_accuracy :  %g,val_loss : %f, val_accuracy :  %g" % (datetime.now(),epoch, train_loss, train_accurate,val_loss,val_accurate)
        with open(train_val_folder+'/'+'train_log.txt',"a+") as f:
            f.write(output+'\n')
            f.close        
        with open(train_val_folder+'/'+'draw_lists.txt',"w") as f:
            f.write('{:}\n {:}\n {:}\n {:}\n '.format(loss_train,acc_train,loss_val,acc_val))
            f.close

        print('[epoch %d] train_loss: %.3f train_acc: %.3f val_loss: %.3f val_acc: %.3f ' %
              (epoch + 1, running_loss / train_steps,train_accurate,val_loss ,val_accurate))
        
        #保存最好的训练、验证权重文件
        # model_folder = './repvggB2g4/save_model/isic{}'.format(time.strftime('%m-%d', time.localtime()))
        model_folder = './save_model/chest/RepVGGB2(1){}'.format(time.strftime('%m-%d', time.localtime()))
        if not os.path.exists(model_folder):
            os.mkdir(model_folder)
        save_path0 = model_folder+'/train{}.pth'.format(time.strftime('%m-%d', time.localtime()))
        save_path1 = model_folder+'/val{}.pth'.format(time.strftime('%m-%d', time.localtime()))        
        if train_accurate > train_best_acc:
            train_best_acc = train_accurate
            torch.save(net.state_dict(),save_path0)        
        if val_accurate > val_best_acc:
            val_best_acc = val_accurate
            torch.save(net.state_dict(),save_path1)
            train_val_folder = './model_train_val/chest/RepVGGB2(1){}'.format(time.strftime('%m-%d', time.localtime()))
            # train_val_folder = './repvggB2g4/model_train_val/isic{}'.format(time.strftime('%m-%d', time.localtime()))
            if not os.path.exists(train_val_folder):
                os.mkdir(train_val_folder)
            with open(train_val_folder+'/targetlist{}.txt'.format(time.strftime('%m-%d %H:%M', time.localtime())),'a+') as f:        
                for tar in targetlist:
                    f.write(str(tar)+'\n')
                    f.close
            with open(train_val_folder+'/predlist{}.txt'.format(time.strftime('%m-%d %H:%M', time.localtime())),'a+') as f:        
                for pre in predlist:
                    f.write(str(pre)+'\n')
                    f.close             
        #评价指标
        acc = accuracy_score(targetlist, predlist, normalize=True)
        F1 = f1_score(targetlist, predlist, average='macro')
        precision = precision_score(targetlist, predlist,  labels=None, pos_label=1, average='macro')
        recall = recall_score(targetlist, predlist,  labels=None,average='macro', sample_weight=None)
        val_lab = label_binarize(targetlist, classes=labels)
        val_pre = label_binarize(predlist, classes=labels)
        auc = roc_auc_score(val_lab, val_pre, average='macro')
        print('acc',acc)
        print('F1',F1)
        print('precision',precision)
        print('recall',recall)
        print('auc',auc)
        # save_path3 = model_folder+'/val_f1{}.pth'.format(time.strftime('%m-%d', time.localtime()))     
        save_path4 = model_folder+'/val_test{}.pth'.format(time.strftime('%m-%d', time.localtime()))     
        save_path5 = model_folder+'/val_auc{}.pth'.format(time.strftime('%m-%d', time.localtime()))     
        # if  F1> val_best_f1:
            # val_best_f1 = F1
            # torch.save(net.state_dict(),save_path3) 
        if acc > 0.95 and acc < 0.958:
            # val_best_pre = precision
            torch.save(net.state_dict(),save_path4) 
        if  auc> val_best_auc:
            val_best_auc = auc
            torch.save(net.state_dict(),save_path5)
        with open(train_val_folder+'/metrics{}.txt'.format(time.strftime('%m-%d', time.localtime())),'a+') as f:        
            f.write('acc:{}, F1:{}, precision:{}, recall:{}, auc:{}\n'.format(acc,F1,precision,recall,auc))
            f.close  
    # output_folder = './repvggB2g4/output/isic{}'.format(time.strftime('%m-%d', time.localtime()))
    output_folder = './output/chest/RepVGGB2(1){}'.format(time.strftime('%m-%d', time.localtime()))
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    matplot_loss(loss_train, loss_val,output_folder)
    matplot_acc(acc_train, acc_val,output_folder)
    print('Finished Training')


if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    main()
