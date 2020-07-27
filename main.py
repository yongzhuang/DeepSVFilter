# coding=utf-8

from __future__ import print_function
import sys
import os
import random
import tensorflow as tf
from PIL import Image
import numpy as np

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import argparse
from glob import glob
from model import *
from bed_process.run import *

parser = argparse.ArgumentParser(description='')

parser.add_argument('--phase', dest='phase', default='train', help='preprocess, train or test')
parser.add_argument('--sv_type', dest='sv_type', default='DEL', help='DEL only')

# preprocess-bed stage
parser.add_argument('--bam_path', dest='bam_path', default='bam/HG002.bam', help='filepath for BAM file')
parser.add_argument('--p_bed_path', dest='p_bed_path', default='bed/P.bed', help='filepath for positive BED')
parser.add_argument('--n_bed_path', dest='n_bed_path', default='bed/N.bed', help='filepath for negative BED')
parser.add_argument('--patch_size', dest='patch_size', type=int, default=224, help='224 or 299')
parser.add_argument('--output_imgs_dir', dest='output_imgs_dir', default='imgs/', help='directory for output image')

# train / test stage
parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=0, help='gpu flag, 1 for GPU and 0 for CPU')
parser.add_argument('--gpu_idx', dest='gpu_idx', default="0", help='GPU idx')
parser.add_argument('--gpu_mem', dest='gpu_mem', type=float, default=0.5, help="0 to 1, gpu memory usage")

parser.add_argument('--model', dest='model_type', default='M1',
                    help='M1(for MobileNet_v1) or M2_1.0(for MobileNet_v2_1.0) or M2_1.4(for MobileNet_v2_1.4) '
                         'or NAS(for NASNet_A_Mobile) or PNAS(for PNASNet_5_Mobile) '
                         'or IR_v2(for Inception_ResNet_v2)')
parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='checkpoint/', help='directory for checkpoints')

# other args for train stage
parser.add_argument('--epoch', dest='epoch', type=int, default=13, help='number of total epoches')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='number of samples in one batch')
parser.add_argument('--start_lr', dest='start_lr', type=float, default=0.001, help='initial learning rate for adam')
parser.add_argument('--eval_every_epoch', dest='eval_every_epoch', default=1,
                    help='evaluating and saving checkpoints every #  epoch')
parser.add_argument('--train_dir', dest='train_dir', default='data/train/', help='directory for training inputs')
parser.add_argument('--eval_dir', dest='eval_dir', default='data/eval/', help='directory for evaluating inputs')
parser.add_argument('--eval_results_dir', dest='eval_results_dir', default='eval_results/',
                    help='directory for evaluating outputs')
parser.add_argument('--summary_dir', dest='summary_dir', default='summary/', help='directory for tensorboard summary')

# other args for test stage
parser.add_argument('--test_dir', dest='test_dir', default='data/test/', help='directory for testing inputs', )
parser.add_argument('--test_results_dir', dest='test_results_dir', default='test_results/',
                    help='directory for testing outputs')

args = parser.parse_args()


def load_images_inception(file, patch_size):
    im = Image.open(file)
    w, h = im.size
    if w != patch_size or h != patch_size:
        im = im.resize((patch_size, patch_size), Image.ANTIALIAS)
    im = np.array(im, dtype="float32") / 127.5 - 1

    # im_c = np.zeros((args.patch_size, args.patch_size, 3), dtype="float32")
    # im_c[:, :, :1] = im[:, :, :1]  # c1 only
    # im_c[:, :, :2] = im[:, :, :2]  # c12 only
    # im_c[:, :, :1] = im[:, :, :1]  # c13 only
    # im_c[:, :, 2:] = im[:, :, 2:]  # c13 only

    return im


def to_train(model, model_name, patch_size):
    if not os.path.exists(os.path.join(args.ckpt_dir, model_name)):
        os.makedirs(os.path.join(args.ckpt_dir, model_name))
    if not os.path.exists(args.eval_results_dir):
        os.makedirs(args.eval_results_dir)

    lr = (args.start_lr / 30.0) * np.ones([args.epoch])
    lr[5:] = args.start_lr / 50.0
    # ********************************************load train data****************************************

    train_p_path = os.path.join(args.train_dir, 'P_list.txt')
    train_p_f = open(train_p_path, 'r')
    train_p_names = [pair_line.rstrip('\n') for pair_line in train_p_f]

    train_n_path = os.path.join(args.train_dir, 'N_list.txt')
    train_n_f = open(train_n_path, 'r')
    train_n_names = [pair_line.rstrip('\n') for pair_line in train_n_f]

    print("[*] Number of training positive data:", len(train_p_names))
    print("[*] Number of training negative data:", len(train_n_names))

    assert len(train_p_names) == len(train_n_names)  # 10000
    print('[*] Number of training data: %d' % (len(train_p_names) + len(train_n_names)))

    train_im = []
    train_label = []

    count = 1
    for line in train_p_names:
        # print("[*] Loading training P image [", count, '/', len(train_p_names), ']', end='\r')
        p_path = line.rstrip('\n')
        p_im = load_images_inception(p_path, patch_size)
        train_im.append(p_im)
        train_label.append([1., 0.])
        count += 1

    count = 1
    for line in train_n_names:
        # print("[*] Loading training N image [", count, '/', len(train_n_names), ']', end='\r')
        n_path = line.rstrip('\n')
        n_im = load_images_inception(n_path, patch_size)
        train_im.append(n_im)
        train_label.append([0., 1.])
        count += 1

    tmp = list(zip(train_im, train_label))
    random.shuffle(tmp)
    train_im, train_label = zip(*tmp)

    # ********************************************load evaluation data****************************************
    eval_p_names, eval_n_names = [], []

    eval_p_path = os.path.join(args.eval_dir, 'P_list.txt')
    eval_p_f = open(eval_p_path, 'r')
    eval_p_names = [pair_line.rstrip('\n') for pair_line in eval_p_f]
    random.shuffle(eval_p_names)

    eval_n_path = os.path.join(args.eval_dir, 'N_list.txt')
    eval_n_f = open(eval_n_path, 'r')
    eval_n_names = [pair_line.rstrip('\n') for pair_line in eval_n_f]
    random.shuffle(eval_n_names)

    print("[*] Number of evaluation positive data:", len(eval_p_names))
    print("[*] Number of evaluation negative data:", len(eval_n_names))

    assert len(eval_p_names) == len(eval_n_names)  # 5000
    print('[*] Number of evaluation data: %d' % (len(eval_p_names) + len(eval_n_names)))

    eval_im = []
    eval_label = []

    count = 1
    for line in eval_p_names:
        # print("[*] Loading evaluation P image [", count, '/', len(eval_p_names), ']', end='\r')
        p_path = line.rstrip('\n')
        p_im = load_images_inception(p_path, patch_size)
        eval_im.append(p_im)
        eval_label.append([1., 0.])
        count += 1

    count = 1
    for line in eval_n_names:
        # print("[*] Loading evaluation N image [", count, '/', len(eval_n_names), ']', end='\r')
        n_path = line.rstrip('\n')
        n_im = load_images_inception(n_path, patch_size)
        eval_im.append(n_im)
        eval_label.append([0., 1.])
        count += 1

    tmp = list(zip(eval_im, eval_label))
    random.shuffle(tmp)
    eval_im, eval_label = zip(*tmp)

    # ********************************************get testing data****************************************

    if args.test_dir == None:
        print("[!] please provide --test_dir")
        exit(0)

    if not os.path.exists(args.test_results_dir):
        os.makedirs(args.test_results_dir)

    model.train(train_im, train_label, eval_im, eval_label,
                batch_size=args.batch_size, patch_size=patch_size,
                epoch=args.epoch, eval_every_epoch=args.eval_every_epoch,
                lr=lr, ckpt_dir=os.path.join(args.ckpt_dir, model_name),
                summary_dir=os.path.join(args.summary_dir, model_name))


def to_test(model, model_name, patch_size):
    if args.test_dir == None:
        print("[!] please provide --test_dir")
        exit(0)

    if not os.path.exists(args.test_results_dir):
        os.makedirs(args.test_results_dir)

    # ********************************************load testing data****************************************

    test_p_path = os.path.join(args.test_dir, 'P_list.txt')
    test_p_f = open(test_p_path, 'r')
    test_p_names = [pair_line.rstrip('\n') for pair_line in test_p_f]
    random.shuffle(test_p_names)

    test_n_path = os.path.join(args.test_dir, 'N_list.txt')
    test_n_f = open(test_n_path, 'r')
    test_n_names = [pair_line.rstrip('\n') for pair_line in test_n_f]
    random.shuffle(test_n_names)

    print("[*] Number of testing positive data:", len(test_p_names))
    print("[*] Number of testing negative data:", len(test_n_names))

    # assert len(test_p_names) == len(test_n_names)  # 5000
    print('[*] Number of testing data: %d' % (len(test_p_names) + len(test_n_names)))

    test_im = []
    test_label = []
    test_names = []

    count = 1
    for line in test_p_names:
        # print("[*] Loading testing P image [", count, '/', len(test_p_names), ']', end='\r')
        p_path = line.rstrip('\n')
        p_im = load_images_inception(p_path, patch_size)
        test_im.append(p_im)
        test_label.append([1., 0.])
        test_names.append(p_path)
        count += 1

    count = 1
    for line in test_n_names:
        print("[*] Loading testing N image [", count, '/', len(test_n_names), ']', end='\r')
        n_path = line.rstrip('\n')
        n_im = load_images_inception(n_path, patch_size)
        test_im.append(n_im)
        test_label.append([0., 1.])
        test_names.append(n_path)
        count += 1

    tmp = list(zip(test_im, test_label, test_names))
    random.shuffle(tmp)
    test_im, test_label, test_names = zip(*tmp)

    model.test(test_im, test_label, test_names, patch_size=patch_size, ckpt_dir=os.path.join(args.ckpt_dir, model_name),
               res_dir=args.test_results_dir)


def main(_):
    _model_name = "MobileNet_v1"
    if args.model_type == "IR_v2":
        _model_name = "Inception_ResNet_v2"
    elif args.model_type == "M1":
        _model_name = "MobileNet_v1"
    elif args.model_type == "M2_1.0":
        _model_name = "MobileNet_v2_1.0"
    elif args.model_type == "M2_1.4":
        _model_name = "MobileNet_v2_1.4"
    elif args.model_type == "NAS":
        _model_name = "NASNet_A_Mobile"
    elif args.model_type == "PNAS":
        _model_name = "PNASNet_5_Mobile"
    else:
        print("input model type", args.model_type, "not exisited")
        exit(1)

    if args.model_type == "IR_v2":
        patch_size = 299
    else:
        patch_size = 224

    if args.use_gpu:
        print("[*] GPU\n")
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_idx
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_mem)
        if args.phase == 'preprocess':
            print("[*] In phase : preprocess")
            trans2img(args.bam_path, args.sv_type, args.p_bed_path, args.n_bed_path, args.output_imgs_dir,
                      args.patch_size)
            print("[*] End of phase : preprocess")
        else:
            with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
                cnn_model = model(sess, is_training=True, model_name=_model_name)
                if args.phase == 'train':
                    print("[*] In phase : train")
                    to_train(cnn_model, _model_name, patch_size)
                elif args.phase == 'test':
                    print("[*] In phase : test")
                    to_test(cnn_model, _model_name, patch_size)
                else:
                    print('[!] Unknown phase')
                    exit(0)
    else:
        print("[*] CPU\n")
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
        if args.phase == 'preprocess':
            print("[*] In phase : preprocess")
            trans2img(args.bam_path, args.sv_type, args.p_bed_path, args.n_bed_path, args.output_imgs_dir,
                      args.patch_size)
            print("[*] End of phase : preprocess")
        else:
            with tf.Session(config=config) as sess:
                cnn_model = model(sess, is_training=True, model_name=_model_name)
                if args.phase == 'train':
                    print("[*] In phase : train")
                    to_train(cnn_model, _model_name, patch_size)
                elif args.phase == 'test':
                    print("[*] In phase : test")
                    to_test(cnn_model, _model_name, patch_size)
                else:
                    print('[!] Unknown phase')
                    exit(0)


if __name__ == '__main__':
    tf.app.run()
