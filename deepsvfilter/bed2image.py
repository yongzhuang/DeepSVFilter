#!/home/huangyalin/software/python3.6.4/bin/python3.6
# encoding: utf-8
'''
@author: hyl
@time: 2019/12/7 15:48
@desc:
'''
import sys
import os
from glob import glob

from deepsvfilter.sv_image import *
#from sv_image_v1 import *

def draw_pic(sam_file, sv_list, mean_size, std_size, output_dir, sv_type, patch_size):

    l_extend, r_extend = patch_size//2, patch_size-patch_size//2
    for i in range(len(sv_list)):
        print("===processing ", i, '/', len(sv_list), "===")
        
        imgs = []
        if sv_type=="DEL":
            drp_list=getDelDRPList(sam_file, sv_list[i], patch_size, mean_size, std_size)
            for flag_LR in [1, 2]:
                bp_position = int(sv_list[i][flag_LR])
                pic_start, pic_end = bp_position - l_extend, bp_position + r_extend
                im = draw_deletion(sam_file, sv_list[i], pic_start, pic_end, flag_LR, drp_list)
                #imgs.append([im, im_re])
                imgs.append(im)
        if sv_type=="DUP":
            drp_list=getTandemDupDRPList(sam_file, sv_list[i], patch_size)
            print(len(drp_list))
            for flag_LR in [1, 2]:
                bp_position = int(sv_list[i][flag_LR])
                pic_start, pic_end = bp_position - l_extend, bp_position + r_extend
                im =draw_tandem_duplication(sam_file, sv_list[i], pic_start, pic_end, flag_LR, drp_list)
                #imgs.append([im, im_re])
                imgs.append(im)
        left_img = imgs[0]
        right_img = imgs[1]
        vertical_im = Image.new('RGB', (patch_size, patch_size), (0, 255, 255))
        vertical_im.paste(left_img, (0, 0))
        vertical_im.paste(right_img, (0, patch_size//2+1))
        save_path = output_dir + '/'+ str(sv_type) + '_' + \
                    ("chr" + sv_list[i][0]) + '_' + str(sv_list[i][1]) + '_' + str(sv_list[i][2]) + ".png"
        vertical_im.save(save_path, "PNG")
    return

def generateImgPathFile(output_dir):
    image_output_dir = os.path.join(output_dir, "image")
    image_output_dir = os.path.abspath(image_output_dir)
    img_path_list = glob(image_output_dir + '/*.png')
    file_path = os.path.join(output_dir, "IMG_PATH.txt")
    f = open(file_path, 'w')
    for path in img_path_list:
        f.write(str(path) + '\n')
    f.flush()
    f.close()
    return

def parse_bed_file(bed_path):
    f = open(bed_path, 'r')
    if not f:
        print('[!] BED Path: [' + bed_path + '] is Empty')
        exit(1)
        return
    sv_list = []
    for line in f:
        line = line.rstrip('\n')
        if len(line.split('\t')) == 1:
            record = line.split(' ')
        else:
            record = line.split('\t')
        sv_list.append((record[0], int(record[1]) + 1, int(record[2]), record[3]))
    return sv_list

def trans2img(bam_path, sv_type, bed_path, output_dir, patch_size, mean_size, std_size):
 
    sv_list = parse_bed_file(bed_path)
    #mean_size, std_size = estimateInsertSizes(bam_path, alignments=1000000)
    
    print("[*] Start generating " + sv_type + " images ===")
    image_output_dir = os.path.join(output_dir, "image")
    if not os.path.exists(image_output_dir):
        os.makedirs(image_output_dir)
    draw_pic(bam_path, sv_list, mean_size, std_size, image_output_dir, sv_type, patch_size)
    generateImgPathFile(output_dir)
    print("[*] End generating " + sv_type + " images ===")


