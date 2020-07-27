# encoding: utf-8
import os
from PIL import Image
from PIL import ImageDraw


def get_rgb(every_pile_record, drp_list):
    # print("====<DEL> Choose RGB Color====")
    r, g, b = 0, 0, 0

    qnames = [a[0] for a in drp_list]
    qtlens = [a[1] for a in drp_list]

    if every_pile_record[2] in ['A', 'T', 'C', 'G']:
        r = 255
    if every_pile_record[4] in qnames:
        index = qnames.index(every_pile_record[4])
        if every_pile_record[-2] == qtlens[index]:  # 'ERR894728.230382826':  # 'ERR903027.256953545':
            g = 255
    if type(every_pile_record[1]) == int and every_pile_record[1] >= 4:
        b = 255

    base_A = tuple([r, g, b])
    # if every_pile_record[4] in qnames:
    #     print((every_pile_record[4], base_A))

    if every_pile_record[4] == 'ERR894728.153408408':
        print("find!!!!!!!!!!!!!!!!!!!!!!!!!! ", every_pile_record)
    return base_A



def draw_pic_column(pos_pic_start, pos_pic_end, pile_record, drp_list, output_dir):
    print("====<DEL> Draw Pictures by Column", (pos_pic_start, pos_pic_end), "====")

    blank = Image.new("RGB", [1600, 1600], "black")

    drawObject = ImageDraw.Draw(blank)

    scale_pix = 5
    blank_pix = 0

    g_set = set()
    for j in range(len(pile_record)):
        x = (pile_record[j][0][0] - pos_pic_start) * scale_pix + blank_pix
        y = (pile_record[j][-1]) * scale_pix + blank_pix

        base_rgb = get_rgb(pile_record[j], drp_list)
        if base_rgb[1] == 255 and pile_record[j][4] not in g_set:
            print("====YELLOW at (", x, ",", y, ")==info:", pile_record[j])
            g_set.add(pile_record[j][4])

        drawObject.rectangle((x, y, x + scale_pix, y + scale_pix), fill=base_rgb)

    # save_dir = "./Generate_Image/chr8_del_test/"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    blank.save(output_dir + 'del_chr8_' + str(pos_pic_start) + '_' + str(pos_pic_start + 320) + "_column.png", "PNG")


def draw_pic_row(sam_file, record, pic_start, pic_end, flag_LR, drp_list, output_dir, sv_type, isPositive, can_type,
                 num):
    lr = ['Left', 'Right']
    pn = ['Negative', 'Positive']
    # print("====Step2: [START]<" + sv_type + "> " + pn[isPositive] + " Draw Pictures by Row" + " " + lr[flag_LR - 1]
    #       + " [" + str(pic_start) + '~' + str(pic_end) + "]====")

    scale_pix = 5
    # blank_pix = 0
    pic_length = (pic_end - pic_start + 1)

    blank = Image.new("RGB", [pic_length * scale_pix, (pic_length // 2) * scale_pix], "black")
    drawObject = ImageDraw.Draw(blank)

    row_range_list = [[-1, -1] for _ in range(pic_length)]
    for read in sam_file.fetch(record[0], pic_start, pic_end):
        read_lr = (read.reference_start + 1, read.reference_end)
        read_range_list = read.get_reference_positions()
        if not read_range_list:
            continue

        height = -1
        for i in range(len(row_range_list)):
            if read_lr[0] > row_range_list[i][1] or read_lr[1] < row_range_list[i][0]:
                row_range_list[i] = [min(read_lr[0], row_range_list[i][0]),
                                     max(read_lr[1], row_range_list[i][1])]
                height = i
                break
        if height == -1:
            print("wrong! no avaliable height============================================")

        r, g, b = 255, 0, 0

        qnames = [a[0] for a in drp_list]
        qtlens = [a[1] for a in drp_list]
        this_qname = read.query_name
        if this_qname in qnames:
            index = qnames.index(this_qname)
            if abs(read.tlen) == qtlens[index]:
                if str(flag_LR) == '1':
                    if int(read_lr[0]) < int(record[1]):
                        g = 255
                else:
                    if int(read_lr[1]) > int(record[2]):
                        g = 255  # (255,255,0)

        # part2
        map_type = 0
        if str(flag_LR) == '1':  #
            #
            if int(read_lr[0]) < int(record[1]):
                ptr = -1
                bp_query_pos = int(record[1]) - read_lr[0] + 1
                for cigar in read.cigartuples:
                    if bp_query_pos > ptr:  #
                        ptr += cigar[1]
                        map_type = cigar[0]
                if bp_query_pos > ptr:
                    map_type = 0
        else:  #
            if int(read_lr[1]) > int(record[2]):  #
                ptr = -1
                bp_query_pos = int(record[2]) - read_lr[0] + 1
                for cigar in read.cigartuples:
                    if bp_query_pos > ptr:
                        ptr += cigar[1]
                        map_type = cigar[0]
        if map_type >= 4:
            b = 255  # (255,0,255)
            print((read.query_name, read_lr[0], read_lr[1], read.cigartuples))

        #
        read_pic_l = (read_lr[0] - pic_start) if read_lr[0] >= pic_start else 0  #
        read_pic_r = (read_lr[1] - pic_start) if read_lr[1] <= pic_end else pic_length - 1

        for x in range(read_pic_l, read_pic_r + 1):
            #
            base_rgb = tuple([r, g, b])
            drawObject.rectangle(
                (x * scale_pix, height * scale_pix, x * scale_pix + scale_pix, height * scale_pix + scale_pix),
                fill=base_rgb)
        drawObject.rectangle(
            (((pic_end - pic_start) // 2) * scale_pix, 0, ((pic_end - pic_start) // 2) * scale_pix,
             pic_length * scale_pix), fill=tuple([0, 255, 255]))
    #     if flag_LR == 2:
    #         if read_lr[0] <= print_pos <= read_lr[1]:
    #             count_coverage += 1
    # print("Pic_Row count_coverage for pos", print_pos, " is ", count_coverage)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if isPositive:
        pn_type = 'pos'
    else:
        pn_type = 'neg'
    save_path = output_dir + can_type + '_' + str(sv_type) + '_' + str(pn_type) + '_' + '#' + num + '_' + (
            "chr" + record[0]) + '_' + lr[flag_LR - 1] + '_' + str(record[flag_LR]) + '_[' + str(pic_start) + ',' + str(
        pic_end) + "]_row.png"
    blank.save(save_path, "PNG")
    # print("=====[DONE]Save to path: ", save_path, "=====")


def draw_pic_statistics(sam_file, record, pic_start, pic_end, flag_LR, drp_list):
    # print("[START]====<" + sv_type + "> " + pn[isPositive] + " Draw Pictures_Statistics" + " " + lr[
    #     flag_LR - 1] + " [" + str(pic_start) + '~' + str(pic_end) + "]====")

    scale_pix = 1
    # blank_pix = 0
    pic_length = (pic_end - pic_start)

    # N*(N//2-1)
    im = Image.new("RGB", [pic_length * scale_pix, (pic_length // 2 - 1) * scale_pix], "black")
    im_draw = ImageDraw.Draw(im)

    column_statistics_list = [[0, 0, 0, 0] for _ in range(pic_length)]

    # print("drp_list=", len(drp_list))
    qnames = [a[0] for a in drp_list]
    # print("len(qnames)=", len(qnames))
    qtlens = [a[1] for a in drp_list]
    # print("len(qtlens)=", len(qtlens))
    rd_count = 0
    for read in sam_file.fetch(record[0], pic_start, pic_end):
        read_range_list = read.get_reference_positions()
        read_lr = (read.reference_start + 1, read.reference_end)
        if not read_range_list:
            continue

        flag_drp = 0

        this_qname = read.query_name
        # print("this_qname=", this_qname)
        if this_qname in qnames:
            index = qnames.index(this_qname)
            if abs(read.tlen) == qtlens[index]:
                if str(flag_LR) == '1':
                    if int(read_lr[0]) < int(record[1]):
                        flag_drp = 1
                else:
                    if int(read_lr[1]) > int(record[2]):
                        flag_drp = 1  # (255,255,0)

        flag_sr = 0
        if str(flag_LR) == '1':  #
            #
            if read_lr[0] < int(record[1]):
                ptr = -1
                bp_query_pos = int(record[1]) - read_lr[0] + 1
                for cigar in read.cigartuples:
                    if bp_query_pos > ptr:  #
                        ptr = ptr + cigar[1] if ptr != -1 else cigar[1]
                        flag_sr = cigar[0]
                if bp_query_pos > ptr:
                    flag_sr = 0
        else:
            if read_lr[1] > int(record[2]):
                ptr = -1
                bp_query_pos = int(record[2]) - read_lr[0] + 1
                for cigar in read.cigartuples:
                    if bp_query_pos > ptr:
                        ptr = ptr + cigar[1] if ptr != -1 else cigar[1]
                        flag_sr = cigar[0]

        #
        read_pic_l = (read_lr[0] - pic_start) if read_lr[0] >= pic_start else 0  #
        read_pic_r = (read_lr[1] - pic_start) if read_lr[1] <= pic_end else pic_length - 1
        # if flag_LR == 2:
        #     print("read.qname=", read.query_name, " read_pic_l=", read_pic_l, "read_pic_r=", read_pic_r,
        #           "column_statistics_list[", print_pos - pic_start, "]=",
        #           column_statistics_list[print_pos - pic_start])
        for i in range(read_pic_l, read_pic_r):
            column_statistics_list[i][0] += 1
            if flag_drp == 1 and flag_sr >= 4:
                column_statistics_list[i][3] += 1
            elif flag_drp == 1:
                column_statistics_list[i][1] += 1
            elif flag_sr >= 4:
                column_statistics_list[i][2] += 1
        rd_count+=1
        if rd_count%10000==0:
            print("rd_count_tmp:", rd_count)
        if rd_count>60000:
            print(record)
            break
    print("break_rd_count=", rd_count)


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

    return im, im.transpose(Image.FLIP_LEFT_RIGHT)
