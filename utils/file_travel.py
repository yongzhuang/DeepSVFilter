# coding=utf-8

import os
import random
from glob import glob
import argparse

parser = argparse.ArgumentParser(description='')

parser.add_argument('--img_p_dir', dest='img_p_dir', default='imgs/P/', help='directory for positive images',
                    required=True)
parser.add_argument('--img_n_dir', dest='img_n_dir', default='imgs/N/', help='directory for negative images',
                    required=True)
parser.add_argument('--output_dir', dest='output_dir', default='data/', help='directory for output file list',
                    required=True)
parser.add_argument('--usage', dest='usage', default='train', help='train or test', required=True)
parser.add_argument('--train_percent', dest='train_percent', default='0.9', help='percentage of training data')

args = parser.parse_args()


#
def getImgFile(img_dir, res_dir, isPositive):
    img_path_list = glob(img_dir + '/*.png')

    pn = ["N", "P"]
    res_path = os.path.join(res_dir, pn[isPositive] + "_list.txt")
    f = open(res_path, 'w')

    #
    for path in img_path_list:
        # record = path.rstrip('\n').split('/')[-1].split('.png')[0].split('_')
        # l = int(record[-2])
        # r = int(record[-1])
        # chr_id = record[-3].lstrip("chr")
        # if abs(r - l) >= 100 and chr_id in [str(i + 1) for i in range(22)]:
        f.write(str(path) + '\n')
    print("test_num:", len(img_path_list))
    f.flush()
    f.close()
    return


#
def splitImgFile(img_dir, train_dir, eval_dir, isPositive, train_percent):
    img_names = glob(img_dir + '/*.png')

    pn = ["N", "P"]
    img_path_list = []

    #
    for path in img_names:
        record = path.rstrip('\n').split('/')[-1].split('.png')[0].split('_')
        l = int(record[-2])
        r = int(record[-1])
        chr_id = record[-3].lstrip("chr")
        if abs(r - l) >= 100 and chr_id in [str(i + 1) for i in range(22)]:
            img_path_list.append(path)

    train_res_path = os.path.join(train_dir, pn[isPositive] + "_list.txt")
    eval_res_path = os.path.join(eval_dir, pn[isPositive] + "_list.txt")
    train_f = open(train_res_path, 'w')
    eval_f = open(eval_res_path, 'w')

    train_list = random.sample(img_path_list, int(len(img_path_list) * train_percent))
    eval_list = [path for path in img_path_list if path not in train_list]
    for path in train_list:
        train_f.write(path + '\n')
    for path in eval_list:
        eval_f.write(path + '\n')
    print("train_num:", len(train_list), " eval_num:", len(eval_list))
    train_f.flush()
    train_f.close()
    eval_f.flush()
    eval_f.close()
    return


def main():
    img_p_dir = args.img_p_dir
    img_n_dir = args.img_n_dir

    print("[*] Starting gathering all filepaths for ", args.usage)
    if args.usage == "test":
        output_dir = os.path.join(args.output_dir, "test")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        getImgFile(img_p_dir, output_dir, isPositive=1)
        getImgFile(img_n_dir, output_dir, isPositive=0)

    elif args.usage == "train":
        train_output_dir = os.path.join(args.output_dir, "train")
        eval_output_dir = os.path.join(args.output_dir, "eval")
        if not os.path.exists(train_output_dir):
            os.makedirs(train_output_dir)
        if not os.path.exists(eval_output_dir):
            os.makedirs(eval_output_dir)
        splitImgFile(img_p_dir, train_output_dir, eval_output_dir, 1, float(args.train_percent))
        splitImgFile(img_n_dir, train_output_dir, eval_output_dir, 0, float(args.train_percent))
    print("[*] Done")


if __name__ == '__main__':
    main()
