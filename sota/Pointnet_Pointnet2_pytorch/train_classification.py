"""
Author: Benny
Date: Nov 2019
"""

import os
import sys
import torch
import numpy as np
import time
import platform #added to check if the device being used is MAC and has metal GPU

import datetime
import logging
import provider
import importlib
import shutil
import argparse

from pathlib import Path
from tqdm import tqdm
from data_utils.ModelNetDataLoader import ModelNetDataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
#from cfg.constants import * 

# This is to use the best available core no matter the device that this is being run on
if platform.system() == "Darwin":  # macOS
    device = "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
else:  # Non-Mac systems (Linux, Windows)
    device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode') # - may not need this but will check with Dr. Zutty
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device') # - may not need this but will check with Dr. Zutty
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    # parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]') - changed so that default would be pointnet2_cls_ssg to be work in conjunction with the llm
    parser.add_argument('--model', default='pointnet2_cls_ssg', help='model name [default: pointnet2_cls_ssg]')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training') # was 200, shortened it to check if it works
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    # -- Additional parser statements - will check to see if they are appropriate, added them from train.py
    parser.add_argument('--data', type=str, help='path to dataset')
    parser.add_argument('--end_lr', type=float, default=0.01, help="stop training when the lr less than end_lr")
    parser.add_argument('--seed', type=int, help='random seed for reproducibility')
    parser.add_argument('--val_r', type=float, default=0.2, help='validation ratio')
    parser.add_argument('--amp', action='store_true', help='use automatic mixed precision')
    return parser.parse_args()


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True


def test(model, loader, num_class=40):
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    classifier = model.eval()

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):

        if not args.use_cpu:
            points, target = points.to(torch.device(device)), target.to(torch.device(device))

        points = points.transpose(2, 1)
        pred, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]

        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1

        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)

    return instance_acc, class_acc


def main(args):
    # ADDED FOR LLM
    '''
    #os.chdir('./sota/Pointnet_Pointnet2_pytorch')
    args = parse_args()
    
    # Import the module dynamically
    pointnet2_cls_ssg_module = importlib.import_module(args.models)
    # Now you can use `pointnet2_cls_ssg_module` to access the contents of `pointnet2_cls_ssg`
    Pointnet_Pointnet2_pytorch = getattr(pointnet2_cls_ssg_module, 'Pointnet_Pointnet2_pytorch')
    get_optimizer = getattr(pointnet2_cls_ssg_module, 'get_optimizer')
    # this will get the gene id value 
    '''
    args = parse_args()

    gene_id = args.model.split('pointnet2_cls_ssg_')[1]
    
    
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('sota/Pointnet_Pointnet2_pytorch/log')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('classification')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    data_path = "/storage/ice-shared/vip-vvk/data/llm_ge_data/modelnet40_normal_resampled"

    train_dataset = ModelNetDataLoader(root=data_path, args=args, split='train', process_data=args.process_data)
    test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test', process_data=args.process_data)
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)

    '''MODEL LOADING'''
    num_class = args.num_category
    model = importlib.import_module(args.model)
    shutil.copy('sota/Pointnet_Pointnet2_pytorch/models/llmge_models/%s.py' % args.model, str(exp_dir))
    shutil.copy('sota/Pointnet_Pointnet2_pytorch/models/pointnet2_utils.py', str(exp_dir))
    shutil.copy('sota/Pointnet_Pointnet2_pytorch/train_classification.py', str(exp_dir))

    classifier = model.GetModel(num_class, normal_channel=args.use_normals)
    criterion = model.GetLoss()
    classifier.apply(inplace_relu)

    if not args.use_cpu:
        classifier = classifier.to(torch.device(device))
        criterion = criterion.to(torch.device(device))


    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0

    '''TRANING'''
    logger.info('Start training...')
    tr_start = time.time()
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        mean_correct = []
        classifier = classifier.train()

        scheduler.step()
        for batch_id, (points, target) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()

            points = points.data.numpy()
            points = provider.random_point_dropout(points)
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points = points.transpose(2, 1)

            if not args.use_cpu:
                   points, target = points.to(torch.device(device)), target.to(torch.device(device))

            pred, trans_feat = classifier(points)
            loss = criterion(pred, target.long(), trans_feat)
            pred_choice = pred.data.max(1)[1]

            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            loss.backward()
            optimizer.step()
            global_step += 1

        train_instance_acc = np.mean(mean_correct)
        log_string('Train Instance Accuracy: %f' % train_instance_acc)

        with torch.no_grad():
            instance_acc, class_acc = test(classifier.eval(), testDataLoader, num_class=num_class)

            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc
            log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
            log_string('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))

            if (instance_acc >= best_instance_acc):
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1

    logger.info('End of training...')
    tr_time = time.time() - tr_start

    #total_params = sum(p.numel() for p in model.parameters())
    results_text = f"{best_instance_acc},{class_acc},{tr_time}"

    filename = f'sota/Pointnet_Pointnet2_pytorch/results/{gene_id}_results.txt'

    dir_path = os.path.dirname(filename)
    # Create the directory, ignore error if it already exists
    os.makedirs(dir_path, exist_ok=True)
    # Open the file in write mode and write the text
    with open(filename, 'w') as file:
        file.write(results_text)
    print(f"results have been written to {filename}")
    print('='*120);print('job done');print('='*120)


if __name__ == '__main__':
    args = parse_args()
    main(args)
