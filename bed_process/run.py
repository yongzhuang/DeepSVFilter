#!/home/huangyalin/software/python3.6.4/bin/python3.6
# encoding: utf-8
'''
@author: hyl
@time: 2019/12/7 15:48
@desc:
'''
import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from bed_process.FILE_CLASS import File
from bed_process.draw import *
from bed_process.function import *


# draw image (deletion, duplication)
def draw_pic(sam_file, del_list, mean_size, std_size, output_dir, sv_type, patch_size, isPositive):
    # [('chr1', 1350092+1, 1351414), ('chr1', 1657013, 1722303)]

    # vcf_list = vcf_list[:10]

    l_extend, r_extend = patch_size//2, patch_size-patch_size//2

    pn = ['N', 'P']
    for i in range(len(del_list)):
        print("===processing ", i, '/', len(del_list), "===")  # , end='\r'

        imgs = []
        for flag_LR in [1, 2]:
            bp_position = int(del_list[i][flag_LR])
            pic_start, pic_end = bp_position - l_extend, bp_position + r_extend

            drp_list, qname_list = getDiscordantReadPairList(sam_file, del_list[i][0], pic_start, pic_end, mean_size,
                                                             std_size, sv_type)

            im, im_re = draw_pic_statistics(sam_file, del_list[i], pic_start, pic_end, flag_LR, drp_list)
            imgs.append([im, im_re])

        left_img = imgs[0][0]
        right_img = imgs[1][0]
        vertical_im = Image.new('RGB', (224, 224), (0, 255, 255))
        vertical_im.paste(left_img, (0, 0))
        vertical_im.paste(right_img, (0, 113))
        save_path = output_dir + del_list[i][-1] + '_' + str(sv_type) + '_' + str(pn[isPositive]) + '_' + \
                    ("chr" + del_list[i][0]) + '_' + str(del_list[i][1]) + '_' + str(del_list[i][2]) + ".png"
        vertical_im.save(save_path, "PNG")

    return


def trans2img(bam_path, sv_type, pos_bed_path, neg_bed_path, output_dir, patch_size):
    file = File()
    bam_file = file.get_bam_file(bam_path)

    pos_list = file.get_bed_file(pos_bed_path)
    neg_list = file.get_bed_file(neg_bed_path)

    mean_size, std_size = getEstimateInsertSizes(bam_file, alignments=1000000)
    # mean_size, std_size = 556.5051524721356, 155.20221350501677
    # mean_size, std_size = 317.9533053876452, 82.33484770981164

    # print("[*] Starting draw positive " + sv_type + " image===")
    # output_p_dir = os.path.join(output_dir, 'P')
    # if not os.path.exists(output_p_dir):
    #     os.makedirs(output_p_dir)
    # draw_pic(bam_file, pos_list, mean_size, std_size, output_p_dir, sv_type, patch_size, isPositive=1)
    # print("[*] End of draw positive image")

    print("[*] Starting draw negative " + sv_type + " image===")
    output_n_dir = os.path.join(output_dir, 'N')
    if not os.path.exists(output_n_dir):
        os.makedirs(output_n_dir)
    draw_pic(bam_file, neg_list, mean_size, std_size, output_n_dir, sv_type, patch_size, isPositive=0)
    print("[*] End of draw negative image")
