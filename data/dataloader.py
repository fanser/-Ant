import os
import numpy as np
import random

import PIL
from PIL import Image

class DBLoader(object):
    def __init__(self, db_path):
        db = np.load(db_path)
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)

        self.imgs = (db['images'] - mean) / std
        self.imgs = self.imgs.astype(np.float32)
        self.labels = db['labels'].astype(np.int)  #0-5
        self.names = db['ids']

        #self.img_h, self.img_w = self.imgs.shape[2], self.imgs.shape[3]
        self.img_h, self.img_w = 256, 256
        self.dset_size = self.imgs.shape[0]

    def iter(self, batch_size):
        assert batch_size > 0
        idxs_epoch = np.arange(self.dset_size)
        np.random.shuffle(idxs_epoch)
        xs = range(0, self.imgs.shape[3] - self.img_w) 
        ys = range(0, self.imgs.shape[2] - self.img_h) 
        for i in xrange(0, self.dset_size, batch_size):
            x1 = random.sample(xs, 1)[0]
            x2 = x1 + self.img_w
            y1 = random.sample(xs, 1)[0]
            y2 = y1 + self.img_h
            #print x1, x2, y1, y2

            idxs =  idxs_epoch[i: i + batch_size]
            #imgs = self.imgs[idxs, :]
            #labels = self.labels[idxs, :]
            imgs = self.imgs[idxs, :, y1:y2, x1:x2]
            labels = self.labels[idxs, y1:y2, x1:x2]
            yield imgs, labels

def get_train_loader():
    train_db_path = '/mnt/cephfs/lab/fzy/LSUN/db/train_384.npz'
    train_loader = DBLoader(train_db_path)
    return train_loader

def get_val_loader():
    val_db_path = '/mnt/cephfs/lab/fzy/LSUN/db/val_384.npz'
    val_loader = DBLoader(val_db_path)
    return val_loader

def get_train_val_db_loader():
    train_loader = get_train_loader()
    val_loader = get_val_loader()
    return train_loader, val_loader

class DataLoader(object):
    def __init__(self, ids):
        from scipy import io

        import torch
        from torchvision import transforms
        img_root = '/mnt/cephfs/lab/fzy/LSUN/images'
        label_root = '/mnt/cephfs/lab/fzy/LSUN/layout_seg'
        img_paths = [os.path.join(img_root, '{}.jpg'.format(id)) for id in ids]
        label_paths = [os.path.join(label_root, '{}.mat'.format(id)) for id in ids]

        self.img_h, self.img_w= 256, 256
        img_tf = transforms.Compose([
            transforms.Resize((self.img_h, self.img_w)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])

        imgs = []
        labels = []
        for label_path, img_path in zip(label_paths, img_paths):
            try:
                img = Image.open(img_path)
                img = img_tf(img)
                if img.shape[0] != 3:
                    raise ValueError

                label = io.loadmat(label_path)['layout'] - 1    #0-4
                label = Image.fromarray(label)
                label = label.resize((self.img_w, self.img_h), resample=PIL.Image.NEAREST)
                label = torch.from_numpy(np.array(label)).long()
            except Exception, e:
                print "Error: img_path ", img_path
                print e
                continue
            imgs.append(img)
            labels.append(label)

        self.imgs = torch.stack(imgs)
        self.labels = torch.stack(labels)
        self.dset_size = self.imgs.shape[0]

    def iter(self, batch_size):
        assert batch_size > 0
        idxs_epoch = np.arange(self.dset_size)
        np.random.shuffle(idxs_epoch)
        for i in xrange(0, self.dset_size, batch_size):
            idxs =  idxs_epoch[i: i + batch_size]
            imgs = self.imgs[idxs, :]
            labels = self.labels[idxs, :]
            yield imgs, labels



class PairVisualization(object):
    def __init__(self, ):
        self.label_to_color = {
                0 : np.array([0, 0, 128]),
                1 : np.array([0, 128, 0]),
                2: np.array([128, 0, 0]),
                3 : np.array([128, 128, 0]),
                4 : np.array([0, 128, 128]),
                5 : np.array([128, 128, 128]),
                }
        self.mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

    def save_img(self, img, path):
        img = img * self.std + self.mean
        img = img.astype(np.uint8)
        img = np.transpose(img, [1,2,0])
        img = Image.fromarray(img) 
        img.save(path)

    def save_label(self, label, path):
        h, w = label.shape
        label = label.astype(np.uint8)
        label_img = np.zeros((h, w, 3), dtype=np.uint8)
        label_img = label_img.reshape(-1, 3)
        label = label.reshape(-1)
        for k, color in self.label_to_color.items():
            label_img[label == k, :] = color.reshape(1, 3)
        label_img = label_img.reshape(h, w, 3)
        label_img = Image.fromarray(label_img)
        label_img.save(path)

    def save_pair(self, img, label, suffix):
        h, w = label.shape
        label_img = np.zeros((h, w, 3), dtype=np.uint8)
        label_img = label_img.reshape(-1, 3)
        label = label.reshape(-1)
        for k, color in self.label_to_color.items():
            label_img[label == k, :] = color.reshape(1, 3)
        label_img = label_img.reshape(h, w, 3)

        img = np.transpose(img, [1,2,0])
        img = Image.fromarray(img) 
        label_img = Image.fromarray(label_img)
        img.save('./img{}.jpg'.format(suffix))
        label_img.save('./label{}.jpg'.format(suffix))

def get_train_val_loader(num_dset=-1):
    train_ids = [line.strip() for line in open('/mnt/cephfs/lab/fzy/LSUN/list/train.txt')][:num_dset]
    val_ids = [line.strip() for line in open('/mnt/cephfs/lab/fzy/LSUN/list/val.txt')][:num_dset]
    train_loader = DataLoader(train_ids)
    val_loader = DataLoader(val_ids)
    return train_loader, val_loader

if __name__ == '__main__':
    train_ids = [line.strip() for line in open('/mnt/cephfs/lab/fzy/LSUN/list/train.txt')][:10]
    val_ids = [line.strip() for line in open('/mnt/cephfs/lab/fzy/LSUN/list/val.txt')][:10]
    train_loader = DataLoader(train_ids)
    val_loader = DataLoader(train_ids)
    '''
    for imgs, labels in train_loader.iter(10):
        print imgs.shape, labels.shape
        print imgs.mean()
        print imgs.max(), labels.max()
        print imgs.min(), labels.min()
        print imgs.dtype, labels.dtype
        print '*'*20
    '''

    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img = train_loader.imgs[4].numpy()
    label = train_loader.labels[4].numpy()
    img = 255 * (img * std + mean)
    img = img.astype(np.uint8)

    saver = PairVisualization() 
    saver.save_pair(img, label, '6')

