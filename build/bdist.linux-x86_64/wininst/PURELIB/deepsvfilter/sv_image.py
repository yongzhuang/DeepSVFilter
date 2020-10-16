# encoding: utf-8
import os
from PIL import Image
from PIL import ImageDraw
import pysam
import numpy as np

def estimateInsertSizes(sam_file_path, alignments=1000000):
    print("==================Estimate Insertion Size==================")
    inserts = []
    count = 0
    sam_file = pysam.AlignmentFile(sam_file_path, "rb")
    for read in sam_file:
        if read.is_proper_pair and read.is_paired  and read.is_read1 and (not read.is_unmapped) and (not read.mate_is_unmapped) and (not read.is_duplicate) and (not read.is_secondary) and (not read.is_supplementary):
            if (read.reference_start < read.next_reference_start and (not read.is_reverse) and read.mate_is_reverse) or (read.reference_start > read.next_reference_start and read.is_reverse and (not read.mate_is_reverse)):
                count += 1
                if count <= alignments:
                   inserts.append(abs(read.tlen))
                else:
                    break
    sam_file.close()
    inserts = sorted(inserts)
    total_num = len(inserts)
    l = int(0.05 * total_num)
    r = int(0.95 * total_num)
    
    inserts = inserts[l:r] 
    insert_mean, insert_std = int(np.mean(inserts)), int(np.std(inserts))
    print("Mean of the insert size is ", insert_mean, "Standard deviation of the insert size is ", insert_std)
    return insert_mean, insert_std

def getDelDRPList(sam_file_path, deletion, patch_size, mean_insert_size, sd_insert_size):
    l_extend, r_extend = patch_size//2, patch_size-patch_size//2
    left_list=[]
    sam_file = pysam.AlignmentFile(sam_file_path, "rb")
    for read in sam_file.fetch(deletion[0], deletion[1]-l_extend-patch_size, deletion[1]+r_extend):
         if read.is_paired and (not read.is_unmapped) and (not read.mate_is_unmapped) and read.reference_start < read.next_reference_start:
             insert_size=abs(read.tlen)
             if (not read.is_reverse) and read.mate_is_reverse and (insert_size - mean_insert_size) > 3 * sd_insert_size:
                 left_list.append(read.qname)
    sam_file.close()
    right_list=[]
    sam_file = pysam.AlignmentFile(sam_file_path, "rb")
    for read in sam_file.fetch(deletion[0], deletion[2]-l_extend, deletion[2]+r_extend+patch_size):
         if read.is_paired and (not read.is_unmapped) and (not read.mate_is_unmapped) and read.reference_start > read.next_reference_start:
             insert_size=abs(read.tlen)
             if read.is_reverse and (not read.mate_is_reverse) and (insert_size - mean_insert_size) > 3 * sd_insert_size:
                 right_list.append(read.qname)
    sam_file.close()
    drplist=list(set(left_list).intersection(set(right_list)))
    return drplist

def getTandemDupDRPList(sam_file_path, duplication, patch_size):
    l_extend, r_extend = patch_size//2, patch_size-patch_size//2
    left_list=[]
    sam_file = pysam.AlignmentFile(sam_file_path, "rb")
    for read in sam_file.fetch(duplication[0], duplication[1]-l_extend, duplication[1]+r_extend+patch_size):
         if read.is_paired and (not read.is_unmapped) and (not read.mate_is_unmapped) and read.reference_start < read.next_reference_start:
             if read.is_reverse and (not read.mate_is_reverse) :
                 left_list.append(read.qname)
    sam_file.close()
    right_list=[]
    sam_file = pysam.AlignmentFile(sam_file_path, "rb")
    for read in sam_file.fetch(duplication[0], duplication[2]-l_extend-patch_size, duplication[2]+r_extend):
         if read.is_paired and (not read.is_unmapped) and (not read.mate_is_unmapped) and read.reference_start > read.next_reference_start:
             if (not read.is_reverse) and read.mate_is_reverse:
                 right_list.append(read.qname)
    sam_file.close()
    print("left=",len(left_list))
    print("right=",len(right_list))
    drplist=list(set(left_list).intersection(set(right_list)))
    print("both=",len(drplist))
    return drplist


def is_left_soft_clipped_read(read): 
    if(read.cigartuples[0][0]==4):
        return True
    else:
        return False

def is_right_soft_clipped_read(read):
    if(read.cigartuples[-1][0]==4):
        return True
    else:
        return False

def draw_deletion(sam_file_path, record, pic_start, pic_end, flag_LR, drp_list):
    
    scale_pix = 1 
    pic_length = (pic_end - pic_start)
    im = Image.new("RGB", [pic_length * scale_pix, (pic_length // 2 - 1) * scale_pix], "black") 
    im_draw = ImageDraw.Draw(im)

    column_statistics_list = [[0, 0, 0, 0] for _ in range(pic_length)]
    sam_file = pysam.AlignmentFile(sam_file_path, "rb")
    for read in sam_file.fetch(record[0], pic_start, pic_end):
        if read.is_unmapped:
            continue
        read_lr = (read.reference_start + 1, read.reference_end)
        
        flag_drp = 0
        if read.qname in drp_list:
            if str(flag_LR) == '1':
                if read.reference_start < read.next_reference_start:
                    flag_drp=1
            else:
                if read.reference_start > read.next_reference_start:
                    flag_drp=1
        
        flag_sr = 0
        if str(flag_LR) == '1':
            if is_right_soft_clipped_read(read):
                flag_sr=1
        else:
            if is_left_soft_clipped_read(read):
                flag_sr=1

        read_pic_l = (read_lr[0] - pic_start) if read_lr[0] >= pic_start else 0
        read_pic_r = (read_lr[1] - pic_start) if read_lr[1] <= pic_end else pic_length - 1
        
        for i in range(read_pic_l, read_pic_r):
            column_statistics_list[i][0] += 1
            if flag_drp == 1 and flag_sr == 1:
                column_statistics_list[i][3] += 1
            elif flag_drp == 1:
                column_statistics_list[i][1] += 1
            elif flag_sr == 1:
                column_statistics_list[i][2] += 1
    sam_file.close()
    for x in range(len(column_statistics_list)):
        y = 0
        rd_count = column_statistics_list[x][0]
        drp_count = column_statistics_list[x][1]
        sr_count = column_statistics_list[x][2]
        both_count = column_statistics_list[x][3]
        
        # SR&RP
        if both_count != 0:
            base_rgb = tuple([255, 255, 255])
            im_draw.rectangle((x * scale_pix, y, x * scale_pix + scale_pix, both_count * scale_pix), fill=base_rgb)
        # split read
        if sr_count != 0:
            base_rgb = tuple([255, 0, 255])
            im_draw.rectangle(
                (x * scale_pix, both_count * scale_pix, x * scale_pix + scale_pix, (both_count + sr_count) * scale_pix),
                fill=base_rgb)
        # discordant read pair
        if drp_count != 0:
            base_rgb = tuple([255, 255, 0])
            im_draw.rectangle(
                (x * scale_pix, (both_count + sr_count) * scale_pix, x * scale_pix + scale_pix,
                 (drp_count + sr_count + both_count) * scale_pix), fill=base_rgb)

        # read depth
        if rd_count != 0:
            base_rgb = tuple([255, 0, 0])
            im_draw.rectangle(
                (x * scale_pix, (drp_count + sr_count + both_count) * scale_pix, x * scale_pix + scale_pix,
                 (rd_count) * scale_pix),
                fill=base_rgb)
    im_draw.rectangle(
        (((pic_end - pic_start) // 2) * scale_pix, 0, ((pic_end - pic_start) // 2) * scale_pix,
         pic_length * scale_pix), fill=tuple([0, 255, 255]))

    #return im, im.transpose(Image.FLIP_LEFT_RIGHT)
    return im

def draw_tandem_duplication(sam_file_path, record, pic_start, pic_end, flag_LR, drp_list):
    
    scale_pix = 1 
    pic_length = (pic_end - pic_start)
    im = Image.new("RGB", [pic_length * scale_pix, (pic_length // 2 - 1) * scale_pix], "black") 
    im_draw = ImageDraw.Draw(im)

    column_statistics_list = [[0, 0, 0, 0] for _ in range(pic_length)]
    sam_file = pysam.AlignmentFile(sam_file_path, "rb")
    for read in sam_file.fetch(record[0], pic_start, pic_end):
        if read.is_unmapped:
            continue
        read_lr = (read.reference_start + 1, read.reference_end)
        
        flag_drp = 0
        if read.qname in drp_list:
            if read.qname in drp_list:
                if str(flag_LR) == '1':
                    if read.reference_start < read.next_reference_start:
                        flag_drp=1
                else:
                    if read.reference_start > read.next_reference_start:
                        flag_drp=1
        flag_sr = 0
        if str(flag_LR) == '1':
            if is_left_soft_clipped_read(read):
                flag_sr=1
                print(read.cigarstring)
        else:
            if is_right_soft_clipped_read(read):
                flag_sr=1
                print(read.cigarstring)

        read_pic_l = (read_lr[0] - pic_start) if read_lr[0] >= pic_start else 0
        read_pic_r = (read_lr[1] - pic_start) if read_lr[1] <= pic_end else pic_length - 1
        
        for i in range(read_pic_l, read_pic_r):
            column_statistics_list[i][0] += 1
            if flag_drp == 1 and flag_sr == 1:
                column_statistics_list[i][3] += 1
            elif flag_drp == 1:
                column_statistics_list[i][1] += 1
            elif flag_sr == 1:
                column_statistics_list[i][2] += 1
    sam_file.close()
    for x in range(len(column_statistics_list)):
        y = 0
        rd_count = column_statistics_list[x][0]
        drp_count = column_statistics_list[x][1]
        sr_count = column_statistics_list[x][2]
        both_count = column_statistics_list[x][3]
        
        # SR&RP
        if both_count != 0:
            base_rgb = tuple([255, 255, 255])
            im_draw.rectangle((x * scale_pix, y, x * scale_pix + scale_pix, both_count * scale_pix), fill=base_rgb)
        # split read
        if sr_count != 0:
            base_rgb = tuple([255, 0, 255])
            im_draw.rectangle(
                (x * scale_pix, both_count * scale_pix, x * scale_pix + scale_pix, (both_count + sr_count) * scale_pix),
                fill=base_rgb)
        # discordant read pair
        if drp_count != 0:
            base_rgb = tuple([255, 255, 0])
            im_draw.rectangle(
                (x * scale_pix, (both_count + sr_count) * scale_pix, x * scale_pix + scale_pix,
                 (drp_count + sr_count + both_count) * scale_pix), fill=base_rgb)

        # read depth
        if rd_count != 0:
            base_rgb = tuple([255, 0, 0])
            im_draw.rectangle(
                (x * scale_pix, (drp_count + sr_count + both_count) * scale_pix, x * scale_pix + scale_pix,
                 (rd_count) * scale_pix),
                fill=base_rgb)
    im_draw.rectangle(
        (((pic_end - pic_start) // 2) * scale_pix, 0, ((pic_end - pic_start) // 2) * scale_pix,
         pic_length * scale_pix), fill=tuple([0, 255, 255]))

    #return im, im.transpose(Image.FLIP_LEFT_RIGHT)
    return im


