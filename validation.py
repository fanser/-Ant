import os
import numpy as np
from data.dataloader import get_val_loader, PairVisualization, get_train_loader
from net.shuffleSeg import LayoutNet

import torch
from torch.autograd import Variable


def run(args):
    model = LayoutNet(args.num_classes)
    model.load_state_dict(torch.load(args.pretrained_model))
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()
    val_loader = get_val_loader()
    #val_loader = get_train_loader()
    print "Data loading done"

    num_val = args.num_val
    imgs = val_loader.imgs[:num_val, :]
    labels = val_loader.labels[:num_val, :]
    names = val_loader.names[:num_val]
    
    with torch.no_grad():
        imgs_v = Variable(torch.from_numpy(imgs))
        labels_v = Variable(torch.from_numpy(labels))

    if torch.cuda.is_available():
        imgs_v = imgs_v.cuda()
        labels_v = labels_v.cuda()

    preds, loss, preds_all = model(imgs_v, labels_v) 
    if torch.cuda.is_available():
        preds = preds.cpu().data.numpy()
        preds_all = preds_all.cpu().data.numpy()
        
    else:
        preds = preds.data.numpy()
        preds_all = preds_all.data.numpy()

    preds = np.argmax(preds, axis=1)
    preds_all = np.argmax(preds_all, axis=2)
    print preds_all.shape

    printer = PairVisualization() 

    for i in range(imgs.shape[0]):
        img = imgs[i, :]
        label = labels[i, :].astype(np.uint8)
        pred = preds[i, :].astype(np.uint8)
        pred1 = preds_all[0, i, :].astype(np.uint8)
        pred2 = preds_all[1, i, :].astype(np.uint8)
        pred3 = preds_all[2, i, :].astype(np.uint8)

        printer.save_img(img, os.path.join(args.save_root, '{}_img.jpg'.format(i)))
        printer.save_label(label, os.path.join(args.save_root, '{}_label.jpg'.format(i)))
        printer.save_label(pred, os.path.join(args.save_root, '{}_pred.jpg'.format(i)))
        printer.save_label(pred1, os.path.join(args.save_root, '{}_pred1.jpg'.format(i)))
        printer.save_label(pred2, os.path.join(args.save_root, '{}_pred2.jpg'.format(i)))
        printer.save_label(pred3, os.path.join(args.save_root, '{}_pred3.jpg'.format(i)))
    print 'Finish Validation'

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--num_classes', type=int, default=6)
    parser.add_argument('--num_val', type=int, default=10)
    parser.add_argument('--save_root', type=str, default='/mnt/cephfs/lab/fzy/LSUN/results')
    #parser.add_argument('--pretrained_model', type=str, default='/mnt/cephfs/lab/fzy/LSUN/models/epoch99.pth')
    parser.add_argument('--pretrained_model', type=str, default='/mnt/cephfs/lab/fzy/LSUN/models/epoch54.pth')
    args = parser.parse_args()
    run(args)
