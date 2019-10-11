
# coding: utf-8

# In[ ]:


import argparse
import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
# from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
# import runpy
import numpy as np
import os
# import cv2
from data import get_train_test_set
# from predict import predict
from data import FaceLandmarksDataset
from Network import *
from PIL import Image, ImageDraw 
import matplotlib.pyplot as plt
import time


# In[ ]:


def model_parameters_init(model):
    '''
    kaiming init
    '''
    for p in model.parameters():
        if len(p.shape) >= 2:
            nn.init.kaiming_normal_(p)
    return model


# In[ ]:


def train(args, train_loader, valid_loader, model, criterion, optimizer, scheduler, device):
    #save model or not
    if args.save_model:
        if not os.path.exists(args.save_directory):
            os.makedirs(args.save_directory)
    
    epochs = args.epochs
    pts_criterion = criterion
    
    train_losses = []
    valid_losses = []
    for epoch_id in range(epochs):
        #monitor training loss
        train_loss = 0.0
        valid_loss = 0.0
        model.train()
        ######################
        #training the model#
        ######################
        train_batch_cnt = 0
        for batch_idx, batch in enumerate(train_loader):
            train_batch_cnt += 1
            img = batch['image']
            landmarks = batch['landmarks']
            
            # groundtruth
            input_img = img.to(device)
            target_pts = landmarks.to(device)
            
            #clear the gradients of all optimized variables
            optimizer.zero_grad()
            
            #get out_put
            #print(input_img.dtype)
            output_pts = model(input_img)
            
            #get loss
            loss = pts_criterion(output_pts, target_pts)
            train_loss += loss.item()
            
            #do bp
            loss.backward()
            optimizer.step()
            
            #show log info
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]  pts_loss: {:.6f}'.format(
                        epoch_id,
                        batch_idx * len(img),
                        len(train_loader.dataset),
                        100. * batch_idx / len(train_loader),
                        loss.item()
                        )
                      )
        #记录train_loss
        train_loss /= train_batch_cnt
        train_losses.append(train_loss)
            
        ######################
        # validate the model #
        ######################
        valid_loss = 0.0
        #change model mode to eval ,not ues BN/Dropout
        model.eval()
        with torch.no_grad():
            valid_batch_cnt = 0
            
            for valid_batch_idx, batch in enumerate(valid_loader):
                valid_batch_cnt += 1
                valid_img = batch['image']
                landmarks = batch['landmarks']
                
                input_img = valid_img.to(device)
                target_pts = landmarks.to(device)
                
                output_pts = model(input_img)
                
                valid_loss_batch = pts_criterion(output_pts, target_pts)
                valid_loss += valid_loss_batch.item()
            
            valid_loss /= valid_batch_cnt * 1.0
            #记录valid_loss
            valid_losses.append(valid_loss)
            print('Valid: pts_loss: {:.6f}'.format(valid_loss))
            #学习率衰减
            scheduler.step()
        print('===========================================================')
        #save model
        if args.save_model and epoch_id % 10 == 0:
            saved_model_name = os.path.join(args.save_directory, 'detector_epoch' + '_' + str(epoch_id) + '.pt')
            torch.save(model.state_dict(), saved_model_name)
    return train_losses, valid_losses


# In[ ]:


def main():
    parser = argparse.ArgumentParser(description='Detector')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--test_batch_size', type=int, default=256, metavar='N',
                        help='input batch size for testing (default: 256)')
    parser.add_argument('--predict_batch_size', type=int, default=1, metavar='N',
                        help='input batch size for predict (default: 1)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=10, metavar='S',
                        help='random seed (default: 10)')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save_model', action='store_true', default=False,
                        help='save the current Model')
    parser.add_argument('--save_directory', type=str, default='trained_models',
                        help='learnt models are saving here')
    parser.add_argument('--phase', type=str, default='Train',   # Train/train, Predict/predict, Finetune/finetune
                        help='training, predicting or finetuning')
    args = parser.parse_args(['--batch_size=64',
                              '--test_batch_size=64',
                              '--predict_batch_size=1',
                              '--epochs=101',
                              '--lr=0.001',
                              '--momentum=0.5',
                              '--seed=1',
                              '--log_interval=10',
                              '--save_model',
                              '--save_directory=trained_models',
                              '--phase=train'])
    ##############################################################################################################
    #设置随机种子
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    #设置CPU/GPU
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    #For multi GPUs, nothing need to change here
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    ###############################################################################################################
    print('===> Loading Datasets')
    train_set, test_set = get_train_test_set()
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size)
    predict_loader = torch.utils.data.DataLoader(test_set, batch_size=args.predict_batch_size)
    ###############################################################################################################
    print('===> Building Model')
    # For single GPU
    print('===> runing on {}'.format(device))
    ###############################################################################################################
    print('===> init model')
    model = Net_Bn()
    model = model_parameters_init(model)
    ###############################################################################################################
    model.to(device)
    criterion_pts = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr= args.lr)
    optimizer = optim.SGD(model.parameters(), lr = args.lr, momentum= args.momentum)
    #学习率衰减
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10 , 0.9)
    ###############################################################################################################
    if args.phase == 'Train' or args.phase == 'train':
        print('===> Start Training')
        train_losses, valid_losses = train(args, train_loader, valid_loader, model, criterion_pts, optimizer, scheduler, device)
        print('===> Done!')
        return train_losses, valid_losses
        
    elif args.phase == 'Test' or args.phase == 'test':
        print('===> Test')
        path_model = os.path.join(args.save_directory, 'detector_epoch' + '_' + str(85) + '.pt')
        model.load_state_dict(torch.load(path_model))
        model = model.to(device)
        model.eval()
        with torch.no_grad():
            valid_batch_cnt = 0
            valid_loss = 0
            for valid_batch_idx, batch in enumerate(valid_loader):
                valid_batch_cnt += 1
                valid_img = batch['image']
                landmarks = batch['landmarks']
                
                input_img = valid_img.to(device)
                target_pts = landmarks.to(device)
#                 print(input_img.shape)
                output_pts = model(input_img)
#                 print(type(output_pts))
                
                valid_loss_batch = criterion_pts(output_pts, target_pts)
                valid_loss += valid_loss_batch.item()
            
            valid_loss /= valid_batch_cnt * 1.0
            print('Valid: pts_loss: {:.6f}'.format(valid_loss))
        print('===> Done!')
        return None, None
        
    elif args.phase == 'Finetune' or args.phase == 'finetune':
        print('===> Finetune')
        path_model = os.path.join(args.save_directory, 'detector_epoch' + '_' + str(50) + '.pt')
        model.load_state_dict(torch.load(path_model))
        model = model.to(device)
        train_losses, valid_losses = train(args, train_loader, valid_loader, model, criterion_pts, optimizer, scheduler, device)
        print('===> Done!')
        return train_losses, valid_losses
        
    elif args.phase == 'Predict' or args.phase == 'predict':
        print('===> Predict')
        path_model = os.path.join(args.save_directory, 'detector_epoch' + '_' + str(50) + '.pt')
        model.load_state_dict(torch.load(path_model))
        model = model.to(device)
        model.eval()
        idx = 99
        with torch.no_grad():
            for i,data in enumerate(predict_loader):
                if i == idx:
                    img = data['image'].to(device)
                    output_pts = model(img)
                    landmarks = output_pts[0].numpy()
                    xs = landmarks[::2]
                    ys = landmarks[1::2]
                    img = transforms.ToPILImage()(img[0].type(torch.uint8))
                    draw = ImageDraw.Draw(img)
                    draw.point(list(zip(xs,ys)),fill = (0))
                    img.show()
                elif i > idx:
                    break
        print('===> Done!')
        return None, None


# In[ ]:


if __name__ == '__main__':
    np.random.seed(1)
    start = time.time()
    train_losses, valid_losses = main()
    end = time.time()
    print('耗时：{}s'.format(end - start))


# In[ ]:


if __name__ == '__main__':
    plt.figure(0,(8,6))
    start = 0
    end = len(train_losses) + 1
    losses_train = train_losses[start:end]
    losses_valid = valid_losses[start:end]
    plt.plot(np.arange(len(losses_train)),losses_train)
    plt.plot(np.arange(len(losses_valid)),losses_valid)
    plt.legend(['train_losses','valid_losses'])
    plt.title('valid_loss:{}'.format(round(losses_valid[-1],2)), fontsize = 15,pad= 15)
#     plt.xlim(8,100)
    plt.xlabel('epochs',fontsize = 15)
    plt.ylabel('loss',fontsize = 15)
    plt.savefig('figure/{}_newnet.jpg'.format(round(losses_valid[-1],2)))
    plt.show()

