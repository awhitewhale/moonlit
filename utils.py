import os, glob, random
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
from PIL import Image
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor
from torch.utils.data import Dataset
import torch
import pytorch_ssim
from torch.autograd import Variable
import scipy.misc
import scipy.io
from os.path import dirname
from os.path import join
import scipy
import PIL
import scipy.ndimage
import numpy as np
import scipy.special
import math

gamma_range = np.arange(0.2, 10, 0.001)
a = scipy.special.gamma(2.0 / gamma_range)
a *= a
b = scipy.special.gamma(1.0 / gamma_range)
c = scipy.special.gamma(3.0 / gamma_range)
prec_gammas = a / (b * c)


def aggd_features(imdata):
    # flatten imdata
    imdata.shape = (len(imdata.flat),)
    imdata2 = imdata * imdata
    left_data = imdata2[imdata < 0]
    right_data = imdata2[imdata >= 0]
    left_mean_sqrt = 0
    right_mean_sqrt = 0
    if len(left_data) > 0:
        left_mean_sqrt = np.sqrt(np.average(left_data))
    if len(right_data) > 0:
        right_mean_sqrt = np.sqrt(np.average(right_data))

    if right_mean_sqrt != 0:
        gamma_hat = left_mean_sqrt / right_mean_sqrt
    else:
        gamma_hat = np.inf
    # solve r-hat norm

    imdata2_mean = np.mean(imdata2)
    if imdata2_mean != 0:
        r_hat = (np.average(np.abs(imdata)) ** 2) / (np.average(imdata2))
    else:
        r_hat = np.inf
    rhat_norm = r_hat * (((math.pow(gamma_hat, 3) + 1) * (gamma_hat + 1)) / math.pow(math.pow(gamma_hat, 2) + 1, 2))

    # solve alpha by guessing values that minimize ro
    pos = np.argmin((prec_gammas - rhat_norm) ** 2);
    alpha = gamma_range[pos]

    gam1 = scipy.special.gamma(1.0 / alpha)
    gam2 = scipy.special.gamma(2.0 / alpha)
    gam3 = scipy.special.gamma(3.0 / alpha)

    aggdratio = np.sqrt(gam1) / np.sqrt(gam3)
    bl = aggdratio * left_mean_sqrt
    br = aggdratio * right_mean_sqrt

    # mean parameter
    N = (br - bl) * (gam2 / gam1)  # *aggdratio
    return (alpha, N, bl, br, left_mean_sqrt, right_mean_sqrt)


def ggd_features(imdata):
    nr_gam = 1 / prec_gammas
    sigma_sq = np.var(imdata)
    E = np.mean(np.abs(imdata))
    rho = sigma_sq / E ** 2
    pos = np.argmin(np.abs(nr_gam - rho));
    return gamma_range[pos], sigma_sq


def paired_product(new_im):
    shift1 = np.roll(new_im.copy(), 1, axis=1)
    shift2 = np.roll(new_im.copy(), 1, axis=0)
    shift3 = np.roll(np.roll(new_im.copy(), 1, axis=0), 1, axis=1)
    shift4 = np.roll(np.roll(new_im.copy(), 1, axis=0), -1, axis=1)

    H_img = shift1 * new_im
    V_img = shift2 * new_im
    D1_img = shift3 * new_im
    D2_img = shift4 * new_im

    return (H_img, V_img, D1_img, D2_img)


def gen_gauss_window(lw, sigma):
    sd = np.float32(sigma)
    lw = int(lw)
    weights = [0.0] * (2 * lw + 1)
    weights[lw] = 1.0
    sum = 1.0
    sd *= sd
    for ii in range(1, lw + 1):
        tmp = np.exp(-0.5 * np.float32(ii * ii) / sd)
        weights[lw + ii] = tmp
        weights[lw - ii] = tmp
        sum += 2.0 * tmp
    for ii in range(2 * lw + 1):
        weights[ii] /= sum
    return weights


def compute_image_mscn_transform(image, C=1, avg_window=None, extend_mode='constant'):
    if avg_window is None:
        avg_window = gen_gauss_window(3, 7.0 / 6.0)
    assert len(np.shape(image)) == 2
    h, w = np.shape(image)
    mu_image = np.zeros((h, w), dtype=np.float32)
    var_image = np.zeros((h, w), dtype=np.float32)
    image = np.array(image).astype('float32')
    scipy.ndimage.correlate1d(image, avg_window, 0, mu_image, mode=extend_mode)
    scipy.ndimage.correlate1d(mu_image, avg_window, 1, mu_image, mode=extend_mode)
    scipy.ndimage.correlate1d(image ** 2, avg_window, 0, var_image, mode=extend_mode)
    scipy.ndimage.correlate1d(var_image, avg_window, 1, var_image, mode=extend_mode)
    var_image = np.sqrt(np.abs(var_image - mu_image ** 2))
    return (image - mu_image) / (var_image + C), var_image, mu_image


def _niqe_extract_subband_feats(mscncoefs):
    # alpha_m,  = extract_ggd_features(mscncoefs)
    alpha_m, N, bl, br, lsq, rsq = aggd_features(mscncoefs.copy())
    pps1, pps2, pps3, pps4 = paired_product(mscncoefs)
    alpha1, N1, bl1, br1, lsq1, rsq1 = aggd_features(pps1)
    alpha2, N2, bl2, br2, lsq2, rsq2 = aggd_features(pps2)
    alpha3, N3, bl3, br3, lsq3, rsq3 = aggd_features(pps3)
    alpha4, N4, bl4, br4, lsq4, rsq4 = aggd_features(pps4)
    return np.array([alpha_m, (bl + br) / 2.0,
                     alpha1, N1, bl1, br1,  # (V)
                     alpha2, N2, bl2, br2,  # (H)
                     alpha3, N3, bl3, bl3,  # (D1)
                     alpha4, N4, bl4, bl4,  # (D2)
                     ])


def get_patches_train_features(img, patch_size, stride=8):
    return _get_patches_generic(img, patch_size, 1, stride)


def get_patches_test_features(img, patch_size, stride=8):
    return _get_patches_generic(img, patch_size, 0, stride)


def extract_on_patches(img, patch_size):
    h, w = img.shape
    patch_size = np.int(patch_size)
    patches = []
    for j in range(0, h - patch_size + 1, patch_size):
        for i in range(0, w - patch_size + 1, patch_size):
            patch = img[j:j + patch_size, i:i + patch_size]
            patches.append(patch)

    patches = np.array(patches)

    patch_features = []
    for p in patches:
        patch_features.append(_niqe_extract_subband_feats(p))
    patch_features = np.array(patch_features)

    return patch_features


def _get_patches_generic(img, patch_size, is_train, stride):
    h, w= np.shape(img)
    if h < patch_size or w < patch_size:
        print("Input image is too small")
        exit(0)

    # ensure that the patch divides evenly into img
    hoffset = (h % patch_size)
    woffset = (w % patch_size)

    if hoffset > 0:
        img = img[:-hoffset, :]
    if woffset > 0:
        img = img[:, :-woffset]

    img = img.astype(np.float32)
    # img2 = scipy.misc.imresize(img, 0.5, interp='bicubic', mode='F')

    im = Image.fromarray(img)
    size = tuple((np.array(im.size) * 0.5).astype(int))
    img2 = np.array(im.resize(size, PIL.Image.BICUBIC))

    mscn1, var, mu = compute_image_mscn_transform(img)
    mscn1 = mscn1.astype(np.float32)

    mscn2, _, _ = compute_image_mscn_transform(img2)
    mscn2 = mscn2.astype(np.float32)

    feats_lvl1 = extract_on_patches(mscn1, patch_size)
    feats_lvl2 = extract_on_patches(mscn2, patch_size / 2)

    feats = np.hstack((feats_lvl1, feats_lvl2))  # feats_lvl3))

    return feats


def niqe(inputImgData):
    patch_size = 96
    module_path = dirname(__file__)

    # TODO: memoize
    params = scipy.io.loadmat(join(module_path, 'data', 'niqe_image_params.mat'))
    pop_mu = np.ravel(params["pop_mu"])
    pop_cov = params["pop_cov"]
    (M, N) = inputImgData.shape

    # assert C == 1, "niqe called with videos containing %d channels. Please supply only the luminance channel" % (C,)
    assert M > (
                patch_size * 2 + 1), "niqe called with small frame size, requires > 192x192 resolution video using current training parameters"
    assert N > (
                patch_size * 2 + 1), "niqe called with small frame size, requires > 192x192 resolution video using current training parameters"

    feats = get_patches_test_features(inputImgData, patch_size)
    sample_mu = np.mean(feats, axis=0)
    sample_cov = np.cov(feats.T)

    X = sample_mu - pop_mu
    covmat = ((pop_cov + sample_cov) / 2.0)
    pinvmat = scipy.linalg.pinv(covmat)
    niqe_score = np.sqrt(np.dot(np.dot(X, pinvmat), X))

    return niqe_score

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        pass


def create_exp_folder():
    path_file_number = glob.glob(pathname='experiment/exp*')
    if len(path_file_number) == 0:
        create_folder('experiment/exp0')
        path_file_number = glob.glob(pathname='experiment/exp*')
    nums = [item[14:] for item in path_file_number]
    if len(glob.glob(pathname='experiment/exp{}/*'.format(max(nums)))) == 0:
        maxexp = max(nums)
        create_folder('experiment/exp{}'.format(maxexp))
    else:
        maxexp = int(max(nums))+1
        create_folder('experiment/exp{}'.format(maxexp))
    return ('experiment/exp{}'.format(maxexp))


def crop_img(image):
    h = image.shape[0]
    w = image.shape[1]
    crop_h = h % 16
    crop_w = w % 16
    return image[crop_h // 2:h - crop_h + crop_h // 2, crop_w // 2:w - crop_w + crop_w // 2, :]


def random_augmentation(*args):
    out = []
    flag_aug = random.randint(1, 7)
    for data in args:
        out.append(data_augmentation(data, flag_aug).copy())
    return out


def data_augmentation(image, mode):
    if mode == 0:
        out = image.numpy()
    elif mode == 1:
        out = np.flipud(image)
    elif mode == 2:
        out = np.rot90(image)
    elif mode == 3:
        out = np.rot90(image)
        out = np.flipud(out)
    elif mode == 4:
        out = np.rot90(image, k=2)
    elif mode == 5:
        out = np.rot90(image, k=2)
        out = np.flipud(out)
    elif mode == 6:
        out = np.rot90(image, k=3)
    elif mode == 7:
        out = np.rot90(image, k=3)
        out = np.flipud(out)
    return out


def crop_patch(opt, img_1, img_2):
    H = img_1.shape[0]
    W = img_1.shape[1]
    ind_H = random.randint(0, H - opt.patch_size)
    ind_W = random.randint(0, W - opt.patch_size)

    patch_1 = img_1[ind_H:ind_H + opt.patch_size, ind_W:ind_W + opt.patch_size]
    patch_2 = img_2[ind_H:ind_H + opt.patch_size, ind_W:ind_W + opt.patch_size]

    return patch_1, patch_2


class TrainDataset(Dataset):
    def __init__(self, opt):
        super(TrainDataset, self).__init__()
        self.opt = opt
        self.counter = 0
        self.path = '/home/data/liuyifan/project/datasets/derain/SPA-Data/' # need be changed while testing
        self.datapath = "/home/data/liuyifan/project/datasets/derain/SPA-Data/real_world.txt" # need be changed while testing
        self.data_item = []
        self.data_item += [item_i.strip() for item_i in open(self.datapath)]
        self.crop_transform = Compose([ToPILImage(), RandomCrop(opt.patch_size), ])
        self.len_data = len(self.data_item)
        self.toTensor = ToTensor()

    def __len__(self):
        return self.len_data

    def __getitem__(self, item):
        imgpath = self.data_item[self.counter]
        img = np.array(Image.open(self.path + imgpath.split(' ')[0]).convert('RGB'))
        croped_img = crop_img(img)
        gt_img = np.array(Image.open(self.path + imgpath.split(' ')[1]).convert('RGB'))
        croped_gtimg = crop_img(gt_img)
        self.counter = (self.counter + 1) % self.len_data
        if self.counter == 0:
            random.shuffle(self.data_item)
        depatch1, gtpatch1 = random_augmentation(*crop_patch(self.opt, croped_img, croped_gtimg))
        depatch2, gtpatch2 = random_augmentation(*crop_patch(self.opt, croped_img, croped_gtimg))
        depatch1, gtpatch1, depatch2, gtpatch2 = self.toTensor(depatch1), self.toTensor(gtpatch1), self.toTensor(depatch2), self.toTensor(gtpatch2)
        return depatch1, depatch2, gtpatch1, gtpatch2


def adjust_learning_rate(epoch, opt, optimizer):
    lr = opt.lr * (0.1 ** (epoch // 60))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def ssim(img1,img2):
    img1 = torch.from_numpy(np.rollaxis(img1, 2)).float().unsqueeze(0)/255.0
    img2 = torch.from_numpy(np.rollaxis(img2, 2)).float().unsqueeze(0)/255.0
    img1 = Variable(img1, requires_grad=False)    # torch.Size([256, 256, 3])
    img2 = Variable(img2, requires_grad=True)
    ssim_value = pytorch_ssim.ssim(img1, img2).item()
    return ssim_value


def psnr(img1, img2):
    mse = np.mean((img1/255. - img2/255.) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
