import os
from os import listdir
import numpy as np
import random
import argparse
import time
import copy
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import torch
from torch.optim import lr_scheduler
import sys
BASE_DIR = os.path.dirname(os.path.abspath('__file__'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
from BiopointData import BiopointDataset
from torch_geometric.data import DataLoader

from net.FocalLoss import focal_loss
from net.brain_networks import Net
from tkinter import _flatten

from util import (normal_transform_train,normal_transform_test,train_val_test_split, sens_spec, site_split,get_index,plot_ROC,
                  write_excel_xlsx, plot_fea, plot_distr)
from mmd_loss import MMD_loss

def dist_loss(s,ratio):
    s = s.sort(dim=1).values
    source = s[:,-int(s.size(1)*ratio):]
    target = s[:,:int(s.size(1)*ratio)]
    res =  MMD_loss()(source,target)
    return -res

def dist_loss2(source,target):
    res =  MMD_loss()(source,target)
    return -res
    
def main():
    
    def train(epoch):
        print('train...........')
        model.train()

        loss_all = 0
        i = 0
        for data in train_loader:

            data = data.to(device)
            optimizer.zero_grad()
            
            output1, score, fea1, fea2 = model(data.x, data.edge_index, data.batch, data.edge_attr, data.pcd)

            loss_fn = focal_loss(alpha=[1,1.5], gamma=2, num_classes=2) # classification loss
            loss_c = loss_fn(output1, data.y)
            loss = loss_c #+ 0.1*loss_f 

            i = i + 1
    
            loss.backward()
            loss_all += loss.item() * data.num_graphs
            # optimzercenter.step()
            optimizer.step()
            scheduler.step()
    
        all_score = score
        return loss_all / len(train_dataset)

    
    ###################### Network Testing Function#####################################
    def test_acc(loader):
        model.eval()
        correct = 0
        preds =[]
        labels = []
        pred_label = []
        for data in loader:
            data = data.to(device)
            output, _, _, _= model(data.x, data.edge_index, data.batch, data.edge_attr, data.pcd)
            pred_y = output.max(dim=1)[1]
            # pred = torch.round(output)
            score = output[:,1].cpu().tolist()
            correct += pred_y.eq(data.y).sum().item()
            labels.append(data.y.tolist())
            pred_label.append(pred_y.tolist())
            preds.append(score)

        return correct / len(loader.dataset), preds,labels, pred_label
    
    
    def test_loss(loader):
        print('testing...........')
        model.eval()
        loss_all = 0
    
        i=0
        for data in loader:
            data = data.to(device)
            output1, fea1, fea2,_ = model(data.x, data.edge_index, data.batch, data.edge_attr, data.pcd)

            loss_fn = focal_loss(alpha=[1,1.5], gamma=2, num_classes=2) # classification loss
            loss_c = loss_fn(output1, data.y)
            loss = loss_c #+ 0.5*loss_fea
            i = i + 1
    
            loss_all += loss.item() * data.num_graphs
        return loss_all / len(loader.dataset)
    
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        
    def visual(loader):
        model.eval()
        features = []
        pcd= []
        labels = []
        outputs = []
        for data in loader:
            data = data.to(device)
            output, feature, _= model(data.x, data.edge_index, data.batch, data.edge_attr, data.pcd)
            feature_array = feature.cpu().detach().numpy()
            X_tsne = tsne.fit_transform(feature_array)
            features.extend(X_tsne)
            outputs.extend(output.cpu().detach().numpy())
            labels.extend(data.y.tolist())
            pcd.extend(data.pcd.cpu().detach().numpy())
        feature = np.array(features)
        outputs = np.array(outputs)
        labels = np.array(labels)
        pcd = np.array(pcd)
        return feature, outputs, labels, pcd
    
    EPS = 1e-15
    device = torch.device("cuda:0")
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=1, help='starting epoch')
    parser.add_argument('--n_epochs', type=int, default=40, help='number of epochs of training')
    parser.add_argument('--ssl_epochs', type=int, default=70, help='number of epochs of training')
    parser.add_argument('--batchSize', type=int, default= 48, help='size of the batches')
    parser.add_argument('--dataroot', type=str,
                        default=r'H:\PHD\learning\research\PRGNN_fMRI-myModify\data\new\train\raw_aal_remove2',
                        help='root directory of the dataset: \
                        {ADHD: E:\OPT\research\fMRI\myModel\comparsion\population-gcn-master\time_correlation\train\nilearn_out_remove\
                         ADHD: E:\OPT\research\fMRI\PRGNN_fMRI-myModify\data\time_correlation\final_train\AAL116Overlap\
                         ADHD: E:\OPT\research\fMRI\PRGNN_fMRI-myModify\data\time_correlation\final_train\\Schaefer2018Overlap\
                         ADHD: E:\OPT\research\fMRI\PRGNN_fMRI-myModify\data\time_correlation\train\nilearn\Schaefer2018remove\
                         ADHD: E:\OPT\research\fMRI\PRGNN_fMRI-myModify\data\time_correlation\train\nilearn\AAL116remove\
                         ADHD: E:\OPT\research\fMRI\PRGNN_fMRI-myModify\data\time_correlation\train\nilearn\Schaefer2018Overlap\
                         ADHD: E:\OPT\research\fMRI\PRGNN_fMRI-myModify\data\time_correlation\train\nilearn\AAL116Overlap\
                         ADHD: E:\OPT\research\fMRI\myMo del\comparsion\population-gcn-master\time_correlation\train\tem\
                         ADHD: E:\OPT\research\fMRI\myModel\comparsion\population-gcn-master\time_correlation\train\nilearn90ROI_remove\
                         ADHD: E:\OPT\research\fMRI\myModel\comparsion\population-gcn-master\time_correlation\train\nilearn_out_remove_ehr\
                         ADNI: E:\OPT\research\fMRI\ADNI_rsfMRI\outlier_remove}')
    parser.add_argument('--testroot', type=str,default=r'H:\PHD\learning\research\PRGNN_fMRI-myModify\data\new\test\raw_aal_remove',
                        help='root directory of the test set')   
    parser.add_argument('--fold', type=int, default=10, help='training which fold')
    parser.add_argument('--lr', type = float, default=0.01, help='learning rate')
    parser.add_argument('--rep', type=int, default=1, help='augmentation times')
    parser.add_argument('--stepsize', type=int, default=20, help='scheduler step size')
    parser.add_argument('--gamma', type=float, default=0.5, help='scheduler shrinking rate')
    parser.add_argument('--weightdecay', type=float, default=5e-2, help='regularization')
    parser.add_argument('--lamb0', type=float, default=1, help='classification loss weight')
    parser.add_argument('--lamb1', type=float, default=1, help='s1 unit regularization')
    parser.add_argument('--lamb2', type=float, default=1, help='s2 unit regularization')
    parser.add_argument('--lamb3', type=float, default=0.1, help='s1 distance regularization')
    parser.add_argument('--lamb4', type=float, default=0.1, help='s2 distance regularization')
    parser.add_argument('--lamb5', type=float, default=0, help='s1 consistence regularization')
    parser.add_argument('--lamb6', type=float, default=0, help='s2 consistence regularization')
    parser.add_argument('--distL', type=str, default='bce', help='bce || mmd')
    parser.add_argument('--poolmethod', type=str, default='None', help='topk || sag || ASA')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam || SGD')
    parser.add_argument('--layer', type=int, default=2, help='number of GNN layers')
    parser.add_argument('--nodes', type=int, default=116, help='{ADHD: number of nodes 161}{ADNI: number of nodes 116}')
    parser.add_argument('--ratio', type=float, default=0.5, help='pooling ratio')
    parser.add_argument('--net', type=str, default='FPT', help='model name')
    parser.add_argument('--shf_indim', type=int, default=100, help='{ADHD: feature dim 161},{ADNI: feature dim 116}')
    parser.add_argument('--indim', type=int, default=90, help='{ADHD: feature dim 161},{ADNI: feature dim 116}')
    parser.add_argument('--aal_indim', type=int, default=116, help='{ADHD: feature dim 161},{ADNI: feature dim 116}')
    parser.add_argument('--nclass', type=int, default=2, help='feature dim')
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--normalization', action='store_true') 
    parser.add_argument('--k', default=16, type=int, help='neighbor num (default:16)')
    parser.add_argument('--conv', default='edge', type=str, help='graph conv layer {edge, gat, DNA, gcn}')
    parser.add_argument('--bias', default=True,  type=bool, help='bias of conv layer True or False')
    parser.add_argument('--norm', default='batch', type=str, help='{batch, instance} normalization')
    parser.add_argument('--numbers', default=871, type=int, 
                        help='numbers of samples: two atlax overlap ADHD:{718} {717}去掉0010044因为gender信息丢失,之前只用了前90roi，这里全部{713}{712}\
                        ADHD:{768 : all samples, 729: outlier delete} ADNI:{173: outlier delete}, 加上test即所有数据共871')
    parser.add_argument('--retrain', default=False, type=bool, help='whether train from used model')   
    parser.add_argument('--model_file', nargs="+", default = r'E:\...\rep1_biopoint_6_FPT_0.pth',
                        type=str, help='wheter train from used model.')    
    parser.add_argument('--ssl', default=False, type=bool, help='whether ssl training')

    # dilated knn
    
    parser.add_argument('--epsilon', default=0.2, type=float, help='stochastic epsilon for gcn')
    parser.add_argument('--stochastic', default=True,  type=bool, help='stochastic for gcn, True or False')
    parser.add_argument('--gae', default=False,  type=bool, help='whether using GAE to predict edge probability')   
    parser.add_argument('--beta', default=1.5, type=float, help='edge probability loss')
    parser.add_argument('--visual', default=False, type=bool, help='whether visulize the feature')

    parser.set_defaults(save_model=True)
    parser.set_defaults(normalization=True)
    opt = parser.parse_args()
    name = 'Biopoint'
    
    ############# Define Dataloader -- need costumize#####################
    datasets = BiopointDataset(opt.dataroot, name)
    # mask = torch.ones(len(datasets), dtype=torch.uint8)
    mask = torch.zeros(len(datasets), dtype=torch.bool)
    site = datasets.data.pcd[:,0].numpy()
    idx = np.where(site != 7)[0]

    mask[idx] = True
    dataset = copy.deepcopy(datasets[mask])
    ########multiple trials
    numbers = list(range(0,10))
        
    for number in numbers:
        tr_Rindex,val_Rindex,te_Rindex = site_split(labels=dataset.data.y[mask].tolist(), pcd = dataset.data.pcd[mask].tolist(), kfold = opt.fold, number = number)
        # tr_Rindex,val_Rindex,te_Rindex = site_split(labels=dataset.data.y.tolist(), pcd = dataset.data.pcd.tolist(), kfold = opt.fold, number = number)
        # tr_Rindex,val_Rindex,te_Rindex = split(labels=dataset.data.y.tolist(), pcd = dataset.data.pcd.tolist(), kfold = opt.fold)
    
        root_fold = 'result/all_epoches/'
        save_number = listdir(root_fold)
        del save_number[-1]
        last_number =max(map(int, save_number))
        n = last_number+1
        save_file = root_fold + str(n) 
        f_file = root_fold+'prediction/' + str(n) + '_' + str(number)
        img_file = 'result/ROC/'+ str(n) + '_' + str(number)
        if not os.path.exists(f_file):
            os.makedirs(f_file)   
        if not os.path.exists(img_file):
            os.makedirs(img_file)    
        if not os.path.exists(save_file):
            os.makedirs(save_file)
            
        all_pred = []
        all_labels = []
    
        for j in range(0,10):        
            if torch.cuda.is_available():
                setup_seed(666) 
                ############### Define Graph Deep Learning Network ##########################
            if opt.net == 'FPT':
                model = Net(opt).to(device)
            if opt.retrain:
                # model = torch.load(opt.model_file)
                checkpoint  = torch.load(opt.model_file[j], map_location=torch.device("cuda:0"))
                # pretrained_dict = model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数
                model_dict = model.state_dict()
                pretrained_dict = {k: v for k, v in checkpoint['net'].items() if k in model_dict}
    
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)
    
            print(model)
                         ##############################################################           

            if opt.optimizer == 'Adam':
                optimizer = torch.optim.Adam(model.parameters(), lr= opt.lr, weight_decay=opt.weightdecay)
            elif opt.optimizer == 'SGD':
                optimizer = torch.optim.SGD(model.parameters(), lr =opt.lr, momentum = 0.9, weight_decay=opt.weightdecay, nesterov = True)
            
            scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.stepsize, gamma=opt.gamma)
                        
            if opt.retrain:
                optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
                scheduler.load_state_dict(checkpoint['scheduler'])  # 加载优化器参数
                # start_epoch = checkpoint['epoch']  # 设置开始的epoch
            ############################### Define Other Loss Functions ########################################
            if opt.distL == 'bce':
                ###### bce loss
                def dist_loss(s,ratio):
                    if ratio > 0.5:
                        ratio = 1-ratio
                    s = s.sort(dim=1).values
                    res =  -torch.log(s[:,-int(s.size(1)*ratio):]+EPS).mean() -torch.log(1-s[:,:int(s.size(1)*ratio)]+EPS).mean()
                    return res
        
            
            ############### split train, val, and test set -- need costumize########################
            tr_index,val_index, te_index = tr_Rindex[j],val_Rindex[j], te_Rindex[j]

            ######################################################################
            train_mask = torch.zeros(len(dataset), dtype=torch.bool)
            test_mask = torch.zeros(len(dataset), dtype=torch.bool)
            val_mask = torch.zeros(len(dataset), dtype=torch.bool)
            train_mask[tr_index] = True
            test_mask[te_index] = True
            val_mask[val_index] = True

            test_dataset = copy.deepcopy(dataset[test_mask])
            train_dataset = copy.deepcopy(dataset[train_mask])
            val_dataset = copy.deepcopy(dataset[val_mask])
            
            
            # ######################## Data Preprocessing ########################
            # ###################### Normalize features ##########################
            if opt.normalization:
                for i in range(train_dataset.data.x.shape[1]):
                    train_dataset.data.x[:, i], lamb, xmean, xstd = normal_transform_train(train_dataset.data.x[:, i])
                    test_dataset.data.x[:, i] = normal_transform_test(test_dataset.data.x[:, i],lamb, xmean, xstd)
                    val_dataset.data.x[:, i] = normal_transform_test(val_dataset.data.x[:, i], lamb, xmean, xstd)
            
            test_loader = DataLoader(test_dataset,batch_size=opt.batchSize,shuffle = False)
            val_loader = DataLoader(val_dataset, batch_size=opt.batchSize, shuffle=False)
            train_loader = DataLoader(train_dataset,batch_size=opt.batchSize, shuffle= True)
            
            print('train_ground_truth: ', train_dataset.data.y[mask][train_mask], 'total: ', len(train_dataset.data.y[mask][train_mask]), 'positive: ',sum(train_dataset.data.y[mask][train_mask]))
            print('ground_truth: ', test_dataset.data.y[mask], 'total: ', len(test_dataset.data.y[mask]), 'positive: ',sum(test_dataset.data.y[mask]))
    
            #######################################################################################
            ############################   Model Training #########################################
            #######################################################################################
            best_model_wts = copy.deepcopy(model.state_dict())
            best_loss = 1e10
            all_result=list()
            
            if opt.visual:
                tsne = TSNE(n_components=2, init='pca', random_state=0, learning_rate=100)
    
                tr_feature, tr_output ,tr_label, tr_pcd = visual(train_loader)
                te_feature, te_output ,te_label, te_pcd = visual(val_loader)
                plot_fea('train', tr_feature, tr_label)
                plot_fea('test', te_feature, te_label)     
                plot_fea('train site', tr_feature, tr_pcd[:,0])
                plot_fea('test site', te_feature, te_pcd[:,0])          
                plot_fea('train gender', tr_feature, tr_pcd[:,1])
                plot_fea('test gender', te_feature, te_pcd[:,1])       
                plot_fea('train age', tr_feature, tr_pcd[:,2])
                plot_fea('test age', te_feature, te_pcd[:,2])
                plot_fea('train hand', tr_feature, tr_pcd[:,3])
                plot_fea('test hand', te_feature, te_pcd[:,3])
                plot_distr('train feature distribution',tr_output ,tr_label)
                plot_distr('test feature distribution', te_output ,te_label)
                return
            
    
            for epoch in range(0, opt.n_epochs):
                since  = time.time()
                tr_loss = train(epoch)
                tr_acc,_,_,_ = test_acc(train_loader)
                val_acc,_,_,_ = test_acc(val_loader)
                val_loss = test_loss(val_loader)
                # te_loss = test_loss(test_loader)
                time_elapsed = time.time() - since
                # test_accuracy, prediction,label,predict_labels = test_acc(test_loader)
    
                print('*====**')
                print('{:.0f}m {:.5f}s'.format(time_elapsed / 60, time_elapsed % 60))
                # print('Epoch: {:03d}, Train Loss: {:.7f},Train Acc: {:.7f}, Val Loss: {:.7f}, Val Acc: {:.7f}, '
                #       'test Loss: {:.7f}, test Acc: {:.7f} '.format(epoch, tr_loss,tr_acc, val_loss, val_acc, te_loss, test_accuracy))
            
                print('Epoch: {:03d}, Train Loss: {:.7f},Train Acc: {:.7f}, Val Loss: {:.7f}, Val Acc: {:.7f}'.format(epoch, tr_loss,tr_acc, val_loss, val_acc))
                if val_loss < best_loss and epoch > 5:
                    print("saving best model")
                    best_loss = val_loss
                    # best_model_wts = copy.deepcopy(model.state_dict())
                    checkpoint = {
                        "net": model.state_dict(),
                        'optimizer':optimizer.state_dict(),
                        'scheduler':scheduler.state_dict(),
                        "epoch": epoch
                    }
                    model_file = 'models/'+str(n)
                    if not os.path.exists(model_file):
                        os.makedirs(model_file)
                    if opt.save_model:
                    #     torch.save(best_model_wts,
                    #                model_file+'/rep{}_biopoint_{}_{}_{}.pth'.format(opt.rep,j,opt.net,opt.lamb5))              
                        torch.save(checkpoint,
                                   model_file+'/rep{}_biopoint_{}_{}_{}.pth'.format(opt.rep,j,opt.net,opt.lamb5))
                        
                all_result.append((tr_loss,tr_acc,val_loss,val_acc))
        
            #######################################################################################
            ######################### Testing on testing set ######################################
            #######################################################################################
    
            write_excel_xlsx(save_file+'/'+str(j)+".xlsx",'all_loss_acc_auc',all_result)
            # val_acc,val_pre,val_label,val_prelabel = test_acc(val_loader)
    
            test_accuracy, prediction,label,predict_labels = test_acc(test_loader)
            te_pre_labels = list(_flatten(predict_labels))
            predictions = list(_flatten(prediction))
            labels = list(_flatten(label))
            # val_label = list(_flatten(val_label))
            # val_preds = list(_flatten(val_pre))
            write_excel_xlsx(f_file+'/'+str(j)+"prediction.xlsx",'prediction',(predictions,te_pre_labels, labels))
            plot_ROC(labels, predictions, img_file, str(j))
    
            # write_excel_xlsx(f_file+'/'+str(j)+"prediction.xlsx",'prediction',(predictions,labels, val_preds, val_label))
            all_labels.extend(labels)
            all_pred.extend(predictions)
            test_l= test_loss(test_loader)
            print("===========================")
            print("Test Acc: {:.7f}, Test Loss: {:.7f} ".format(test_accuracy, test_l))
            print(opt)
            sens, spec = sens_spec(te_pre_labels,labels)
            # val_sens, val_spec = sens_spec(val_prelabel[0],val_label)
    
            torch.cuda.empty_cache()
            f = open(save_file+'/ACC_SPE_SEN_each_epoch.txt','a')
            f.write('\n te_acc:{},spe{},sen:{}'.format(test_accuracy,spec,sens))
            # f.write('\n val_acc:{},spe{},sen:{}'.format(val_acc,val_spec,val_sens))
            f.close()        
            f = open(save_file+'/split{}.txt'.format(str(j)),'a')
            f.write('\n tr_index:{},\n val_index{},\n te_index:{}'.format(tr_index,val_index, te_index))
            f.close()
        fpr, tpr, thresholds_keras = roc_curve(all_labels, all_pred)
        Fauc = auc(fpr, tpr)
        print("AUC : ", Fauc)
        plt.figure()
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr, label='area = {:.3f}'.format(Fauc))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best') 
        f = open(save_file+'/ACC_SPE_SEN_each_epoch.txt','a')
        f.write('\n model:{}'.format(str(model)))
        f.write('\n parameter:{}'.format(str(opt)))
        f.close()
        plt.savefig(img_file+"/ROC_10x_over.png")
        plt.show()
        
    return

if __name__ == '__main__':
    main()    
    
