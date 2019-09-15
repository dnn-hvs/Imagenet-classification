import argparse
import os
import random
import shutil
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import xlwt
from vgg import vgg11, vgg13, vgg16, vgg19
from resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from alexnet import AlexNet
from squeezenet import SqueezeNet1_0, SqueezeNet1_1
from densenet import densenet121, densenet161, densenet169, densenet201
from inceptionv3 import inception_v3
from googlenet import googlenet

models = {
    'alexnet': AlexNet,
    'vgg11': vgg11,
    'vgg13': vgg13,
    'vgg16': vgg16,
    'vgg19': vgg19,
    'sqnet1_0': SqueezeNet1_0,
    'sqnet1_1': SqueezeNet1_1,
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
    'densenet121': densenet121,
    'densenet161': densenet161,
    'densenet169': densenet169,
    'densenet201': densenet201,
    'googlenet': googlenet,
    'inception': inception_v3
}

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--model_dir', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--save_file', default='', type=str, metavar='PATH', required=True,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')


def main():
    wb = xlwt.Workbook()
    sheet = wb.add_sheet('Accuracies')
    style = xlwt.easyxf('font: bold 1')
    sheet.write(0, 0, 'Model Name', style)
    sheet.write(0, 1, 'Acc @ 1', style)
    sheet.write(0, 2, 'Acc @ 5', style)
    args = parser.parse_args()
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.model_dir:
        if os.path.isfile(args.model_dir):
            raise NotADirectoryError
        row = 1
        for x in os.listdir(args.model_dir):
            arch = x.split('_')[0]
            if arch == 'sqnet1':
                arch += x.split('_')[1]
            # create model
            # if args.pretrained:
            #     print("=> using pre-trained model '{}'".format(arch))
            #     # model = models[arch](pretrained=True)
            #     if arch == 'alexnet' or arch == 'sqnet1_1' or arch == 'sqnet1_0':
            #         model = models[arch]()
            #     else:
            #         model = models[arch](pretrained=True)
            # else:
            print("=> creating model '{}'".format(arch))
            model = models[arch]()
            print(model.state_dict().keys())

            model_pth = os.path.join(args.model_dir, x)

            print("=> loading checkpoint '{}'".format(model_pth))
            checkpoint = torch.load(model_pth)
            args.start_epoch = checkpoint['epoch']
            state_dict_ = checkpoint['state_dict']
            state_dict = {}

            # convert data_parallal to model
            for k in state_dict_:
                if k.startswith('module') and not k.startswith('module_list'):
                    state_dict[k[7:]] = state_dict_[k]
                else:
                    state_dict[k[8:]] = state_dict_[k]
            model_state_dict = model.state_dict()

            # check loaded parameters and created model parameters
            for k in state_dict:
                if k in model_state_dict:
                    if state_dict[k].shape != model_state_dict[k].shape:
                        print('Skip loading parameter {}, required shape{}, '
                              'loaded shape{}.'.format(
                                  k, model_state_dict[k].shape, state_dict[k].shape))
                        state_dict[k] = model_state_dict[k]
                else:
                    print('Drop parameter {}.'.format(k))
            for k in model_state_dict:
                if not (k in state_dict):
                    print('No param {}.'.format(k))
                    state_dict[k] = model_state_dict[k]
                    model.load_state_dict(state_dict, strict=False)
                    # model.load_state_dict(checkpoint['state_dict'])
                    # optimizer.load_state_dict(checkpoint['optimizer'])
                    print("=> loaded checkpoint '{}' (epoch {})"
                          .format(model_pth, checkpoint['epoch']))
                else:
                    print("=> no checkpoint found at '{}'".format(model_pth))

            torch.cuda.set_device(args.gpu)
            model = model.cuda(args.gpu)

            valdir = os.path.join(args.data, 'val')
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

            val_loader = torch.utils.data.DataLoader(
                datasets.ImageFolder(valdir, transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ])),
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True)

            acc1, acc5 = validate(val_loader, model, args)
            sheet.write(row, 0, x)
            sheet.write(row, 1, acc1)
            sheet.write(row, 1, acc5)
            row += 1

        wb.save(args.save_file + '.xlsx')

    else:
        print('''You call me and not tell me what do I work with? Me leavin!\nNext time, pass in the model_dir, if ya don't mind''')
        return


def validate(val_loader, model, args):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for (images, target) in range(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)[-1]
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target)
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1, 5)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        # print(pred)
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
