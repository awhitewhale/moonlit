import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
from torch.utils.data import DataLoader
from utils import *
from moonlit_pro import moonlit
from torchvision.utils import save_image
import cv2
from PIL import Image
import tqdm


def test():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dataset = TrainDataset()
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=False,
                            pin_memory=True,
                            drop_last=False,
                            num_workers=0)
    net = torch.load('experiment/exp0/best_epoch.pth')
    net = net.cuda()
    net = torch.nn.DataParallel(net, device_ids=[0,1,2,3,4,5,6,7])
    net.eval()

    for query, key, query_gt, key_gt in tqdm.tqdm(dataloader):
        query, key, query_gt, key_gt = query.to(device), key.to(device), query_gt.to(device), key_gt.to(device)
        restored, query, labels = net.module(x_query=query, x_key=key)
        save_image(restored, 'restored.jpg')
        save_image(query_gt, 'query_gt.jpg')
        restored_j = cv2.imread('restored.jpg')
        query_gt_j = cv2.imread('query_gt.jpg')
        img_ssim = ssim(restored_j, query_gt_j)
        img_psnr = psnr(restored_j, query_gt_j)
        img_niqe = np.array(Image.open('restored.jpg').convert('LA'))[:, :, 0]
        img_niqe = niqe(img_niqe)
        print(img_psnr, img_ssim, img_niqe)



if __name__ == '__main__':
    test()

