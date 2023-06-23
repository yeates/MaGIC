import cv2
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
import numpy as np
import glob
# import pyspng
import PIL.Image
from tqdm import tqdm
import torch
import dnnlib as dnnlib
import scipy.linalg
import sklearn.svm
import json
from PIL import Image


_feature_detector_cache = dict()

# load annotations
with open('examples/captions_val2017.json', 'r', encoding='utf-8') as fp: 
    data = json.load(fp)['annotations']
prompt_dic = {}
for file in data:
    name = "%012d.jpg" % file['image_id']
    prompt_dic[name] = file['caption']
    
# import
from transformers import AutoProcessor, AutoModel

# load pickscore model
device = "cuda"
processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"

processor = AutoProcessor.from_pretrained(processor_name_or_path)
model = AutoModel.from_pretrained(model_pretrained_name_or_path).eval().to(device)


def get_feature_detector(url, device=torch.device('cpu'), num_gpus=1, rank=0, verbose=False):
    assert 0 <= rank < num_gpus
    key = (url, device)
    if key not in _feature_detector_cache:
        is_leader = (rank == 0)
        if not is_leader and num_gpus > 1:
            torch.distributed.barrier() # leader goes first
        with dnnlib.util.open_url(url, verbose=(verbose and is_leader)) as f:
            _feature_detector_cache[key] = torch.jit.load(f).eval().to(device)
        if is_leader and num_gpus > 1:
            torch.distributed.barrier() # others follow
    return _feature_detector_cache[key]

class FeatureStats:
    def __init__(self, capture_all=False, capture_mean_cov=False, max_items=None):
        self.capture_all = capture_all
        self.capture_mean_cov = capture_mean_cov
        self.max_items = max_items
        self.num_items = 0
        self.num_features = None
        self.all_features = None
        self.raw_mean = None
        self.raw_cov = None

    def set_num_features(self, num_features):
        if self.num_features is not None:
            assert num_features == self.num_features
        else:
            self.num_features = num_features
            self.all_features = []
            self.raw_mean = np.zeros([num_features], dtype=np.float64)
            self.raw_cov = np.zeros([num_features, num_features], dtype=np.float64)

    def is_full(self):
        return (self.max_items is not None) and (self.num_items >= self.max_items)

    def append(self, x):
        x = np.asarray(x, dtype=np.float32)
        assert x.ndim == 2
        if (self.max_items is not None) and (self.num_items + x.shape[0] > self.max_items):
            if self.num_items >= self.max_items:
                return
            x = x[:self.max_items - self.num_items]

        self.set_num_features(x.shape[1])
        self.num_items += x.shape[0]
        if self.capture_all:
            self.all_features.append(x)
        if self.capture_mean_cov:
            x64 = x.astype(np.float64)
            self.raw_mean += x64.sum(axis=0)
            self.raw_cov += x64.T @ x64

    def append_torch(self, x, num_gpus=1, rank=0):
        assert isinstance(x, torch.Tensor) and x.ndim == 2
        assert 0 <= rank < num_gpus
        if num_gpus > 1:
            ys = []
            for src in range(num_gpus):
                y = x.clone()
                torch.distributed.broadcast(y, src=src)
                ys.append(y)
            x = torch.stack(ys, dim=1).flatten(0, 1) # interleave samples
        self.append(x.cpu().numpy())

    def get_all(self):
        assert self.capture_all
        return np.concatenate(self.all_features, axis=0)

    def get_all_torch(self):
        return torch.from_numpy(self.get_all())

    def get_mean_cov(self):
        assert self.capture_mean_cov
        mean = self.raw_mean / self.num_items
        cov = self.raw_cov / self.num_items
        cov = cov - np.outer(mean, mean)
        return mean, cov

    def save(self, pkl_file):
        with open(pkl_file, 'wb') as f:
            pickle.dump(self.__dict__, f)

    @staticmethod
    def load(pkl_file):
        with open(pkl_file, 'rb') as f:
            s = dnnlib.EasyDict(pickle.load(f))
        obj = FeatureStats(capture_all=s.capture_all, max_items=s.max_items)
        obj.__dict__.update(s)
        return obj


def read_image(image_path):
    with open(image_path, 'rb') as f:
        if False and image_path.endswith('.png') and pyspng is not None:
            image = pyspng.load(f.read())
        else:
            image = np.array(PIL.Image.open(f).convert('RGB').resize((512,512)))
    if image.ndim == 2:
        image = image[:, :, np.newaxis] # HW => HWC
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    # image = image.transpose(2, 0, 1) # HWC => CHW

    return image



def calculate_metrics(l1, l2, is_pick):
    assert(len(l1) == len(l2)), (len(l1), len(l2))
    print('length:', len(l1))

    # build detector
    detector_url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt'
    detector_kwargs = dict(return_features=True) # Return raw features before the softmax layer.
    device = torch.device('cuda:0')
    detector = get_feature_detector(url=detector_url, device=device, num_gpus=1, rank=0, verbose=False)
    detector.eval()

    stat1 = FeatureStats(capture_all=True, capture_mean_cov=True, max_items=len(l1))
    stat2 = FeatureStats(capture_all=True, capture_mean_cov=True, max_items=len(l1))
    
    pickscore = []
    with torch.no_grad():
        for fpath1, fpath2 in tqdm(zip(l1, l2)):
            _, name1 = os.path.split(fpath1)
            _, name2 = os.path.split(fpath2)
            name1 = name1.split('.')[0]
            name2 = name2.split('.')[0]
            # assert name1 == name2, 'Illegal mapping: %s, %s' % (name1, name2)

            img1 = read_image(fpath1).astype(np.float64)
            img2 = read_image(fpath2).astype(np.float64)
            assert img1.shape == img2.shape, 'Illegal shape'
            fea1 = detector(torch.from_numpy(img1.transpose(2, 0, 1)).unsqueeze(0).to(torch.uint8).to(device), **detector_kwargs)
            stat1.append_torch(fea1, num_gpus=1, rank=0)
            fea2 = detector(torch.from_numpy(img2.transpose(2, 0, 1)).unsqueeze(0).to(torch.uint8).to(device), **detector_kwargs)
            stat2.append_torch(fea2, num_gpus=1, rank=0)
            
            if is_pick:
                pickscore.append(cal_PickScore(fpath1, fpath2))
    
    # calculate fid
    mu1, sigma1 = stat1.get_mean_cov()
    mu2, sigma2 = stat2.get_mean_cov()
    m = np.square(mu1 - mu2).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma1, sigma2), disp=False) # pylint: disable=no-member
    fid = np.real(m + np.trace(sigma1 + sigma2 - s * 2))

    # calculate pids and uids
    fake_activations = stat1.get_all()
    real_activations = stat2.get_all()
    svm = sklearn.svm.LinearSVC(dual=False)
    svm_inputs = np.concatenate([real_activations, fake_activations])
    svm_targets = np.array([1] * real_activations.shape[0] + [0] * fake_activations.shape[0])
    print('SVM fitting ...')
    svm.fit(svm_inputs, svm_targets)
    uids = 1 - svm.score(svm_inputs, svm_targets)
    real_outputs = svm.decision_function(real_activations)
    fake_outputs = svm.decision_function(fake_activations)
    pids = np.mean(fake_outputs > real_outputs)

    return fid, uids, pids, np.mean(np.array(pickscore)) if len(pickscore) else -1.


def cal_PickScore(imgpath1, imgpath2):

    def calc_probs(prompt, images):
        
        # preprocess
        image_inputs = processor(
            images=images,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(device)
        
        text_inputs = processor(
            text=prompt,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(device)


        with torch.no_grad():
            # embed
            image_embs = model.get_image_features(**image_inputs)
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
        
            text_embs = model.get_text_features(**text_inputs)
            text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
        
            # score
            scores = model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
            
            # get probabilities if you have multiple images to choose from
            probs = torch.softmax(scores, dim=-1)
        
        return probs.cpu().tolist()

    pil_images = [Image.open(imgpath1), Image.open(imgpath2)]
    prompt = prompt_dic[os.path.basename(imgpath2)]
    pickscore = calc_probs(prompt, pil_images)[0]
    print(f'{pickscore:.4}', prompt, os.path.basename(imgpath1))
    return pickscore



def calculate_metrics_std(folder1, folder2, is_pick, cal_std):
    l1 = sorted(glob.glob(folder1 + '/*.png') + glob.glob(folder1 + '/*.jpg'))
    l2 = sorted(glob.glob(folder2 + '/*.png') + glob.glob(folder2 + '/*.jpg'))
    assert (len(l1) % len(l2) == 0), (len(l1), len(l2))
    print('length:', len(l1))
    
    if cal_std:
        var = len(l1) // len(l2)
        assert var > 1, var
        var_l1 = [[] for _ in range(var)]
        for f in l1:
            idx = int(f.split('.')[0].split('_')[-1])
            var_l1[idx].append(f)
        fid_arr, uid_arr, pid_arr, pick_arr = [], [], [], []
        for idx in range(var):
            fid, uids, pids, pickscore = calculate_metrics(sorted(var_l1[idx]), l2, is_pick)
            fid_arr.append(fid)
            uid_arr.append(uids)
            pid_arr.append(pids)
            pick_arr.append(pickscore)
        return np.mean(fid_arr), np.mean(uid_arr), np.mean(pid_arr), np.mean(pick_arr), \
            np.std(fid_arr), np.std(uid_arr), np.std(pid_arr), np.std(pick_arr)
    else:
        return *calculate_metrics(l1, l2, is_pick), 0, 0, 0, 0



if __name__ == '__main__':
    cal_std = False
    dataset = 'coco'
    folder1 = f'magicdata/quantitative/seg/mat_{dataset}/'
    folder2 = f'magicdata/gt/rgb/{dataset}/'
    print(folder1, folder2)
    
    fid, uids, pids, pickscore, fid_std, uid_std, pid_std, pick_std = \
        calculate_metrics_std(folder1, folder2, is_pick = dataset=='coco', cal_std=cal_std)
    print('done')
    
    with open('metrics.txt', 'a') as f:
        f.write('> folder1: ' + folder1 + '\n')
        f.write('fid: %.4f, pids: %.4f, uids: %.4f, pickscore: %.4f \n' % (fid, pids, uids, pickscore))
        f.write('fid_std: %.4f, pid_std: %.4f, uid_std: %.4f, pick_std: %.4f \n' % (fid_std, pid_std, uid_std, pick_std))