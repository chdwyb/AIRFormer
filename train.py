import sys
import time
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
# 自定义
from utils import torchPSNR, MixUp_AUG
from datasets import *
from options import Options
from AIRFormer import AIRFormer
from loss import dct_layer


if __name__ == '__main__':

    opt = Options()
    cudnn.benchmark = True
    cnt = 0
    best_psnr = 0

    random.seed(opt.Seed)
    torch.manual_seed(opt.Seed)
    torch.cuda.manual_seed(opt.Seed)
    torch.manual_seed(opt.Seed)
    EPOCH = opt.Epoch
    best_epoch = 0
    BATCH_SIZE_TRAIN = opt.Batch_Size_Train
    BATCH_SIZE_VAL = opt.Batch_Size_Val
    PATCH_SIZE_TRAIN = opt.Patch_Size_Train
    PATCH_SIZE_VAL = opt.Patch_Size_Val
    LEARNING_RATE = opt.Learning_Rate

    inputPathTrain = opt.Input_Path_Train
    targetPathTrain = opt.Target_Path_Train
    inputPathVal = opt.Input_Path_Val
    targetPathVal = opt.Target_Path_Val

    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()
    DCT = dct_layer()
    if opt.CUDA_USE:
        criterion_mse = criterion_mse.cuda()
        DCT = DCT.cuda()

    myNet = AIRFormer()
    myNet = nn.DataParallel(myNet)
    if opt.CUDA_USE:
        myNet = myNet.cuda()

    optimizer = optim.AdamW(myNet.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    ######### Scheduler ###########
    scheduler = CosineAnnealingLR(optimizer, EPOCH, eta_min=1e-7)

    datasetTrain = MyTrainDataSet(inputPathTrain, targetPathTrain, patch_size=PATCH_SIZE_TRAIN)
    trainLoader = DataLoader(dataset=datasetTrain, batch_size=BATCH_SIZE_TRAIN, shuffle=True,
                             drop_last=True, num_workers=opt.Num_Works, pin_memory=True)

    datasetValue = MyValueDataSet(inputPathVal, targetPathVal, patch_size=PATCH_SIZE_VAL)
    valueLoader = DataLoader(dataset=datasetValue, batch_size=BATCH_SIZE_VAL, shuffle=True,
                             drop_last=True, num_workers=opt.Num_Works, pin_memory=True)

    # print(torch.load('./pretrained/model_best_.pth'))
    if opt.Pre_Train:
        if os.path.exists('pretrained/model_best.pth'):
            if opt.CUDA_USE:
                myNet.load_state_dict(torch.load('./pretrained/model_best.pth'))
                print('The pretrained model is loaded!')
            else:
                myNet.load_state_dict(torch.load('./pretrained/model_best.pth', map_location=torch.device('cpu')))
        print('-------------------------------------------------------------------------------------------------------')

    for epoch in range(EPOCH):
        myNet.train()
        iters = tqdm(trainLoader, file=sys.stdout)
        epochLoss = 0
        timeStart = time.time()
        for index, (x, y) in enumerate(iters, 0):

            myNet.zero_grad()
            optimizer.zero_grad()

            if opt.CUDA_USE:
                input_train, target = Variable(x).cuda(), Variable(y).cuda()
            else:
                input_train, target = Variable(x), Variable(y)

            if epoch > 50:
                 input_train, target = MixUp_AUG().aug(input_train, target)
            output_train = myNet(input_train)

            loss_fre = criterion_mse(DCT(output_train), DCT(target))
            loss = F.smooth_l1_loss(output_train, target) + 0.04 * loss_fre

            loss.backward()
            optimizer.step()
            epochLoss += loss.item()

            iters.set_description('Training !!!  Epoch %d / %d,  Batch Loss %.6f' % (epoch+1, EPOCH, loss.item()))

        if epoch%3==0:
            myNet.eval()
            psnr_val_rgb = []
            for index, (x, y) in enumerate(valueLoader, 0):
                input_, target_value = (x.cuda(), y.cuda()) if opt.CUDA_USE else (x, y)
                with torch.no_grad():
                    output_value = myNet(input_)
                for output_value, target_value in zip(output_value, target_value):
                    psnr_val_rgb.append(torchPSNR(output_value, target_value))

            psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()

            if psnr_val_rgb >= best_psnr:
                best_psnr = psnr_val_rgb
                best_epoch = epoch
                torch.save(myNet.state_dict(), 'pretrained/model_best.pth')

        scheduler.step()
        cnt += 1
        if epoch%20==0:
            torch.save(myNet.state_dict(), f'./pretrained/model_{cnt}.pth')
        timeEnd = time.time()
        print("----------------------------------------------------------------------------")
        print("Epoch:  {}  Finished,  Time:  {:.4f} s,  Loss:  {:.6f}, lr:  {:.8f}, psnr:  {:.3f}, best_psnr:  {:.3f}.".format(epoch+1, timeEnd-timeStart, epochLoss, scheduler.get_last_lr()[0], psnr_val_rgb, best_psnr))
        print('-------------------------------------------------------------------------------------------------------')
    print("Training Process Finished ! Best Epoch : {} , Best PSNR : {:.2f}".format(best_epoch, best_psnr))





