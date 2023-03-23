import argparse, os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import *
from moonlit_pro import moonlit
from torchvision.utils import save_image
import cv2
import pandas as pd
from PIL import Image
import tqdm


def train(opt):
    result = pd.DataFrame(columns=('epoch', 'total_iter', 'loss', 'PSNR', 'SSIM', 'NIQE'))
    total_iter = 0
    min_loss = 114514
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    save_path = create_exp_folder()
    dataset = TrainDataset(opt)
    dataloader = DataLoader(dataset,
                            batch_size=opt.batch_size,
                            shuffle=True,
                            pin_memory=True,
                            drop_last=True,
                            num_workers=opt.num_workers)
    net = moonlit(opt).cuda()
    net = torch.nn.DataParallel(net, device_ids=[0,1,2,3,4,5,6,7])
    net.train()
    optimizer = optim.Adam(net.parameters(), lr=opt.lr)
    UMRC_loss = nn.CrossEntropyLoss().cuda()
    LDRN_loss = nn.L1Loss().cuda()

    for epoch in range(opt.epoch):
        for query, key, query_gt, key_gt in tqdm.tqdm(dataloader):
            query, key, query_gt, key_gt = query.to(device), key.to(device), query_gt.to(device), key_gt.to(device)
            optimizer.zero_grad()
            if epoch < opt.UMRC_epoch:
                key_moco, query_moco, labels_moco, inter_moco = net.module.UMRC(query, key, opt)
                contrast_loss = UMRC_loss(query_moco, labels_moco,)
                loss = contrast_loss
            else:
                restored, query, labels = net.module(x_query=query, x_key=key, opt=opt)
                contrast_loss_value = UMRC_loss(query, labels)
                LDRN_loss_value = LDRN_loss(restored, query_gt)
                loss = LDRN_loss_value + 0.1 * contrast_loss_value
            total_iter = total_iter + 1
            loss.backward()
            optimizer.step()
        if epoch >= opt.UMRC_epoch:
            save_image(restored, 'restored.jpg')
            save_image(query_gt, 'query_gt.jpg')
            restored_j = cv2.imread('restored.jpg')
            query_gt_j = cv2.imread('query_gt.jpg')
            img_ssim = ssim(restored_j, query_gt_j)
            img_psnr = psnr(restored_j, query_gt_j)
            img_niqe = np.array(Image.open('restored.jpg').convert('LA'))[:, :, 0]
            img_niqe = niqe(img_niqe)
            df = [epoch+1, total_iter, loss.item(), img_psnr, img_ssim, img_niqe]
            result.loc[total_iter] = df
            result.to_csv('result.csv')
        if (epoch + 1) % 50 == 0:
            torch.save(net.state_dict(), save_path + '/epoch_' + str(epoch + 1) + '.pth')
        adjust_learning_rate(epoch, opt, optimizer)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--UMRC_epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--patch_size', type=int, default=128)
    parser.add_argument('--net_name', type=str, default='resnet')
    # parser.add_argument('--net_name', type=str, default='vit')
    parser = parser.parse_args()
    train(parser)

