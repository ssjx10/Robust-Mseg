import os
import os.path as osp
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torchsummary import summary
from torchvision.transforms import Compose
from torch.utils.data import Dataset, DataLoader
import torch.nn.init as init
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from model import RobustMseg
from transform import transforms, SegToMask
from loss import DiceLoss, WeightedCrossEntropyLoss, GeneralizedDiceLoss, task_loss, KL_divergence
from metrics import MeanIoU, DiceCoefficient, DiceRegion
from evaluation import eval_overlap
from BraTSdataset import GBMset
from utils import seed_everything, init_weights

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"
parallel = True

if __name__ == '__main__':
    
    '''dataload'''
    seed = 20
    seed_everything(seed)
    pat_num = 285
    x_p = np.zeros(pat_num,)
    # target value
    y_p = np.zeros(pat_num,)
    indices = np.arange(pat_num)
    x_train_p, x_test_p, y_train_p, y_test_p, idx_train, idx_test = train_test_split(x_p, y_p, indices, test_size=0.2, random_state=20)
    x_train_p, x_valid_p, y_train_p, y_valid_p, idx_train, idx_valid = train_test_split(x_train_p, y_train_p, idx_train, test_size=1/8, random_state=20)

    train_batch = 8
    crop_size = 112
    valid_batch = 10
    trainset = GBMset(sorted(idx_train), transform=transforms(shift=0.1, flip_prob=0.5, random_crop=crop_size), m_full=True, lazy=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch,
                                              shuffle=True, drop_last=True, num_workers=4, pin_memory=True)

    validset = GBMset(sorted(idx_valid), transform=transforms(random_crop=crop_size), m_full=True, lazy=True)
    validloader = torch.utils.data.DataLoader(validset, batch_size=valid_batch,
                                              shuffle=False, num_workers=2, pin_memory=True)

    ov_trainset = GBMset(sorted(idx_train), transform=transforms(), lazy=True)
    ov_trainloader = torch.utils.data.DataLoader(ov_trainset, batch_size=1,
                                              shuffle=False, num_workers=4)

    ov_validset = GBMset(sorted(idx_valid), transform=transforms(), lazy=True)
    ov_validloader = torch.utils.data.DataLoader(ov_validset, batch_size=1,
                                              shuffle=False, num_workers=4)
    ov_testset = GBMset(sorted(idx_test), transform=transforms(), lazy=True)
    ov_testloader = torch.utils.data.DataLoader(ov_testset, batch_size=1,
                                              shuffle=False, num_workers=4)

    '''model setting'''
    model = RobustMseg()
    model.apply(init_weights)
    if parallel:
        model = nn.DataParallel(model)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    num_epochs= 360
    print_every = 20
    validate_every = 20
    overlapEval_every = 160
    dir_name = 'model'

    learning_rate = 0.0001
    weight_decay = 0.00001
    alpha = 0.1 # for kl loss
    train_loss, train_dice = [], []
    valid_loss, valid_dice = [], []

    dice_loss = DiceLoss()
    l1_loss = nn.L1Loss()
    l2_loss = nn.MSELoss()
    dc = DiceCoefficient()
    dcR = DiceRegion()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    lambda1 = lambda epoch: (1 - epoch / num_epochs)**0.9
    sch = lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1])
    
    for i in range(num_epochs):
        epoch_loss = 0.0
        tr_dice = 0.0
        tr_wt_dice = 0.0
        tr_tc_dice = 0.0
        tr_ec_dice = 0.0

        model.train()
        start_perf_counter = time.perf_counter()
        start_process_time = time.process_time()
        for x_batch, x_m_batch, mask_batch, _ in trainloader:
            x_batch = x_batch.float().to('cuda')
            x_m_batch = x_m_batch.float().to('cuda')
            mask_batch = mask_batch.float().to('cuda')

            outputs, recon_outputs, mu, sigma = model(x_m_batch)

            # (1) Update G network about mask + MVAE + GAN
            wce, dice = task_loss(outputs, mask_batch)
            recon = l2_loss(recon_outputs, x_batch)
            kl = 0.0
            for m in range(4): # modality
                kl += KL_divergence(mu[m], torch.log(torch.square(sigma[m])))
                
            loss = wce + dice + alpha*recon*4 + alpha*kl

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            avg_dice = dc(outputs.detach(), mask_batch.detach())
            wt_dice = dcR(outputs.detach(), mask_batch.detach())
            tc_dice = dcR(outputs.detach(), mask_batch.detach(), 'TC')
            ec_dice = dcR(outputs.detach(), mask_batch.detach(), 'EC')
            tr_dice += avg_dice.item()
            tr_wt_dice += wt_dice.item()
            tr_tc_dice += tc_dice.item()
            tr_ec_dice += ec_dice.item()

        perf_counter = time.perf_counter() - start_perf_counter
        process_time = time.process_time() - start_process_time
        epoch_loss /= len(trainloader)
        tr_dice /= len(trainloader)
        tr_wt_dice /= len(trainloader)
        tr_tc_dice /= len(trainloader)
        tr_ec_dice /= len(trainloader)

        train_loss.append(epoch_loss)
        train_dice.append(tr_dice)

        va_loss = 0.0
        va_dice = 0.0
        va_wt_dice = 0.0
        va_tc_dice = 0.0
        va_ec_dice = 0.0
        va_wt_dice_m = 0.0
        va_tc_dice_m = 0.0
        va_ec_dice_m = 0.0

        if i<5 or (i + 1) % validate_every == 0:
            with torch.no_grad():
                model.eval()
                # Valid accuracy
                for x_batch, x_m_batch, mask_batch, _ in validloader:

                    x_batch = x_batch.float().to('cuda')
                    x_m_batch = x_m_batch.float().to('cuda')
                    mask_batch = mask_batch.long().to('cuda')
                    pred, _, _, _ = model(x_batch)
                    pred_m, _, _, _ = model(x_m_batch)
                    dice = dice_loss(pred, mask_batch)
                    loss = dice

                    va_loss += loss.item()
                    avg_dice = dc(pred.detach(), mask_batch.detach())
                    wt_dice = dcR(pred.detach(), mask_batch.detach())
                    tc_dice = dcR(pred.detach(), mask_batch.detach(), 'TC')
                    ec_dice = dcR(pred.detach(), mask_batch.detach(), 'EC')
                    wt_dice_m = dcR(pred_m.detach(), mask_batch.detach())
                    tc_dice_m = dcR(pred_m.detach(), mask_batch.detach(), 'TC')
                    ec_dice_m = dcR(pred_m.detach(), mask_batch.detach(), 'EC')

                    va_dice += avg_dice.item()
                    va_wt_dice += wt_dice.item()
                    va_tc_dice += tc_dice.item()
                    va_ec_dice += ec_dice.item()
                    va_wt_dice_m += wt_dice_m.item()
                    va_tc_dice_m += tc_dice_m.item()
                    va_ec_dice_m += ec_dice_m.item()


                va_loss /= len(validloader)
                va_dice /= len(validloader)
                va_wt_dice /= len(validloader)
                va_tc_dice /= len(validloader)
                va_ec_dice /= len(validloader)
                va_wt_dice_m /= len(validloader)
                va_tc_dice_m /= len(validloader)
                va_ec_dice_m /= len(validloader)

                valid_loss.append(va_loss)
                valid_dice.append(va_dice)

        if i == 0:
            print(f'perf_counter per epoch : {time.strftime("%H:%M:%S", time.gmtime(perf_counter))}')
            print(f'process_time per epoch : {time.strftime("%H:%M:%S", time.gmtime(process_time))}')

        if i<5 or (i + 1) % print_every == 0:
            print('Epoch [{}/{}], Train_Loss: {:.4f}, Train_dice: {:.4f}, Train_wt_dice: {:.4f}, Train_tc_dice: {:.4f}, Train_ec_dice: {:.4f},\
                  \nValid_Loss: {:.4f}, Valid_dice: {:.4f}, Valid_wt_dice: {:.4f}, Valid_tc_dice: {:.4f}, Valid_ec_dice: {:.4f},\
                  \nValid_wt_dice: {:.4f}, Valid_tc_dice: {:.4f}, Valid_ec_dice: {:.4f}'
                  .format(i + 1, num_epochs, epoch_loss, tr_dice, tr_wt_dice, tr_tc_dice, tr_ec_dice,
                          va_loss, va_dice, va_wt_dice, va_tc_dice, va_ec_dice,
                         va_wt_dice_m, va_tc_dice_m, va_ec_dice_m))

        if (i + 1) == num_epochs or (i + 1) % 20 == 0:
            print(eval_overlap(ov_testloader, model, patch_size=crop_size, overlap_stepsize=crop_size//2, batch_size=valid_batch, num_classes=3))
        if (i + 1) == num_epochs or (i + 1) % overlapEval_every == 0:
            print(eval_overlap(ov_validloader, model, patch_size=crop_size, overlap_stepsize=crop_size//2, batch_size=valid_batch, num_classes=3))
        if (i + 1) == num_epochs or (i + 1) % overlapEval_every == 0:
            print(eval_overlap(ov_trainloader, model, patch_size=crop_size, overlap_stepsize=crop_size//2, batch_size=valid_batch, num_classes=3))
        if (i+1) >= 160 and (i + 1) % 10 == 0:
            save_dir = dir_name + '/'
            if parallel:
                m = model.module
            else:
                m = model
            torch.save(m.state_dict(), save_dir + str(i+1) + '.pth')#, _use_new_zipfile_serialization=False)

        sch.step()

