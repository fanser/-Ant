from data.dataloader import get_train_val_loader, get_train_val_db_loader
from net.shuffleSeg import LayoutNet

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


def loop(loader, model, batch_size, epoch, interval, optimizer=None):
    for i, (imgs, labels) in enumerate(loader.iter(batch_size)):
        imgs = Variable(torch.from_numpy(imgs))
        labels = Variable(torch.from_numpy(labels))
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            labels = labels.cuda()
        
        preds, loss, _ = model(imgs, labels) 
        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if i % interval == 0:
            print "Epoch {}, Iter {}/{}, Loss {}".format(epoch, i, loader.dset_size / batch_size, loss.data.cpu()[0])

def run(args):
    model = LayoutNet(args.num_classes)
    if torch.cuda.is_available():
        model = model.cuda()
    #train_loader, val_loader = get_train_val_loader(10)
    train_loader, val_loader = get_train_val_db_loader()
    print "Data loading done"
    for epoch in range(args.epochs):
        lr = args.lr * args.decay ** epoch
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=args.momentum)
        loop(train_loader, model, args.train_batch_size, epoch, args.train_interval, optimizer)
        loop(val_loader, model, args.val_batch_size, epoch, args.val_interval)
        if torch.cuda.is_available():
            torch.save(model.cpu().state_dict(), args.model_path + 'epoch{}.pth'.format(epoch))
            model = model.cuda()
        else:
            torch.save(model.state_dict(), args.model_path + 'epoch{}.pth'.format(epoch))
    print 'Finish Training'

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--decay', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--num_classes', type=int, default=6)
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--val_batch_size', type=int, default=128)
    parser.add_argument('--train_interval', type=int, default=10)
    parser.add_argument('--val_interval', type=int, default=1)
    parser.add_argument('--model_path', type=str, default='/mnt/cephfs/lab/fzy/LSUN/models/')
    args = parser.parse_args()
    run(args)


