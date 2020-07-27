# encoding: utf-8
import numpy as np


#
def getEstimateInsertSizes(full_sam_file, alignments=2000000):
    print("==================Estimate Insertion Size==================")

    #
    inserts = []
    count = 0
    for read in full_sam_file.head(2 * alignments):
        if read.is_paired and (not read.mate_is_unmapped) and read.tlen:
            count += 1
            if count < alignments:
                inserts.append(abs(read.tlen))

    #
    inserts = sorted(inserts)
    total_num = len(inserts)
    l = int(0.0001 * total_num)
    r = int(0.9999 * total_num)
    inserts = inserts[l:r]

    print("total_num =", total_num)
    print("l = ", l, ", r= ", r)

    #
    i = 0
    for i in range(len(inserts)):
        if abs(inserts[i]) > 2000:
            print("find reads.tlen > 2000 at pos", i, "break")
            break
    inserts = inserts[:i]

    #
    insert_mean, insert_std = np.mean(inserts), np.std(inserts)
    print("insert_mean = ", insert_mean, " insert_std = ", insert_std)
    print("accept_range = (", (insert_mean - 3 * insert_std), "~", (insert_mean + 3 * insert_std), ")")

    if insert_mean > 10000 or insert_std > 1000 or insert_mean < 1 or insert_std < 1:
        print('''WARNING: anomalous insert sizes detected. Please 
              double check or consider setting values manually.''')

    return insert_mean, insert_std


#
def getDiscordantReadPairList(sam_file, chr_id, pic_start, pic_end, mean_size, std_size, sv_type):
    # print("====Step 1: <" + sv_type + "> Get Discordant Read Pair(RP) List" + " [" + str(pic_start) + '~' + str(
    #     pic_end) + "]====")

    drp_res = []
    qname_res = []
    if sv_type == 'DEL':  #
        for read in sam_file.fetch(chr_id, pic_start, pic_end):
            # poses = list(read.get_reference_positions())
            # print((read.query_name, read.reference_start, read.reference_start + read.query_length,read.is_proper_pair, read.is_paired,
            #        read.is_read1, read.is_read2, read.is_reverse, read.mate_is_reverse, read.mate_is_unmapped, read.is_paired and not read.mate_is_unmapped))

            qname = read.query_name
            insert_size = abs(read.tlen)
            # if qname == 'HISEQ1:28:HA2RLADXX:1:1108:8467:37043':
            #     print('HISEQ1:28:HA2RLADXX:1:1108:8467:37043 is_proper_pair=', read.is_proper_pair)
            #     print('HISEQ1:28:HA2RLADXX:1:1108:8467:37043 is_paired=', read.is_paired)
            #     print('HISEQ1:28:HA2RLADXX:1:1108:8467:37043 is_read1=', read.is_read1)
            #     print('HISEQ1:28:HA2RLADXX:1:1108:8467:37043 is_read2=', read.is_read2)
            #     print('HISEQ1:28:HA2RLADXX:1:1108:8467:37043 is_reverse=', read.is_reverse)
            #     print('HISEQ1:28:HA2RLADXX:1:1108:8467:37043 mate_is_reverse=', read.mate_is_reverse)
            #     print('HISEQ1:28:HA2RLADXX:1:1108:8467:37043 mate_is_unmapped=', read.mate_is_unmapped)
            # ['HISEQ1:22:H9UJNADXX:1:1116:20356:31963', 1351414, 'F2R1', 1652]
            # ['HISEQ1:25:H9UD6ADXX:1:1201:4322:23679', 1351445, 'F2R1', 1818]
            # ['HISEQ1:28:HA2RLADXX:1:2206:3599:49548', 1351452, 'F2R1', 1710]
            # ['HISEQ1:28:HA2RLADXX:2:2112:17714:79167', 1351455, 'F2R1', 2000]
            # ['HISEQ1:28:HA2RLADXX:1:2211:14933:73777', 1351467, 'F2R1', 1770]
            # ['HISEQ1:21:H9V1VADXX:2:1213:7148:79799', 1351511, 'F2R1', 1788]
            # ['HISEQ1:19:H8VDAADXX:1:2108:2139:93519', 1351517, 'F2R1', 1804]
            # ['HISEQ1:18:H8VC6ADXX:1:1211:16017:30133', 1351527, 'F2R1', 1903]
            # ['HISEQ1:29:HA2WPADXX:1:1107:16091:46752', 1351537, 'F2R1', 1920]
            # ['HISEQ1:24:H9TKDADXX:1:2210:9023:39116', 1351565, 'F2R1', 1874]
            if read.is_paired and (not read.mate_is_unmapped) and insert_size > 0:
                flag_FR = None
                if abs(insert_size - mean_size) > 3 * std_size:
                    # F?R?
                    # F1R2
                    if read.is_read1 and not read.is_reverse and read.mate_is_reverse:  # F1
                        if read.reference_start < read.next_reference_start:  #
                            flag_FR = 'F1R2'
                            # if qname == 'ERR894727.125333570':
                            #     print(qname, "is F1R2")
                    elif read.is_read2 and read.is_reverse and not read.mate_is_reverse:  # R2
                        if read.reference_start > read.next_reference_start:  #
                            flag_FR = 'F1R2'
                            # if qname == 'ERR894727.125333570':
                            #     print(qname, "is F1R2")
                    # F2R1
                    elif read.is_read1 and read.is_reverse and not read.mate_is_reverse:  # R1
                        if read.reference_start > read.next_reference_start:  #
                            flag_FR = 'F2R1'
                            # if qname == 'ERR894727.125333570':
                            #     print(qname, "is F2R1")
                    elif read.is_read2 and not read.is_reverse and read.mate_is_reverse:  # F2
                        if read.reference_start < read.next_reference_start:  #
                            flag_FR = 'F2R1'
                            # if qname == 'ERR894727.125333570':
                            #     print(qname, "is F2R1")
                    if flag_FR:
                        drp_res.append([qname, read.reference_start, flag_FR, insert_size])
    # print("drp_res=", drp_res, " len(drp_res)= ", len(drp_res))
    drp_list = list(set((a[0], a[-1]) for a in drp_res))
    # print("drp_list=", drp_list)
    return drp_list, qname_res


#
def getPileupRecord(sam_file, chr_id, pos_pic_start, pos_pic_end, pos_bp, flag_LR):
    print("====<DEL> Get Pile-up Read Records", (pos_pic_start, pos_pic_end), "====")

    pile_record = []
    last_pile_column_record = []

    # print_pos = 55763915
    # flag = False
    qname_list = []

    for pile_column in sam_file.pileup(chr_id, pos_pic_start, pos_pic_end):
        # if not pos_pic_start <= pile_column.pos <= pos_pic_end:
        #     continue
        pile_column_record = []
        height = -1

        if pile_column.pos == 55763900:
            for pile_read in pile_column.pileups:
                qname_list.append(pile_read.alignment.query_name)
        for pile_read in pile_column.pileups:  #
            height += 1
            this_qname = pile_read.alignment.query_name
            # if this_qname == 'ERR894728.153408408':
            #     print("'ERR894728.153408408.tlen", abs(pile_read.alignment.tlen))
            # if this_qname == 'ERR894728.215565856':
            #     print("'ERR894728.215565856.tlen", abs(pile_read.alignment.tlen))
            # if this_qname == 'ERR894727.125333570':
            #     print("'ERR894727.125333570.tlen", abs(pile_read.alignment.tlen))

            cigar_str = pile_read.alignment.cigarstring

            map_type = 0

            cur_ref_pos = pile_column.pos
            bp_ref_pos = pos_bp
            cur_query_pos = pile_read.query_position
            bp_query_pos = 0

            if str(flag_LR) == '1':
                if cur_ref_pos > bp_ref_pos:
                    map_type = 0
                else:
                    ptr = 0
                    bp_query_pos = (bp_ref_pos - cur_ref_pos + 1) + cur_query_pos
                    for cigar in pile_read.alignment.cigartuples:
                        if bp_query_pos >= ptr:
                            ptr += cigar[1]
                            map_type = cigar[0]

            else:
                if cur_ref_pos < bp_ref_pos:
                    map_type = 0
                else:
                    ptr = 0
                    bp_query_pos = cur_query_pos - (cur_ref_pos - bp_ref_pos + 1)
                    for cigar in pile_read.alignment.cigartuples:
                        if bp_query_pos >= ptr:
                            ptr += cigar[1]
                            map_type = cigar[0]

            this_qname = pile_read.alignment.query_name
            last_qname_list = [r[4] for r in last_pile_column_record]

            if this_qname in last_qname_list:
                new_height = last_pile_column_record[last_qname_list.index(this_qname)][-1]
                # if pile_column.pos == print_pos:  # or pile_column.pos == 55764006:
                #     print("appeared in last column! height =", new_height)
                while new_height > height:
                    # this_column.append(('None', height))
                    pile_result = ((cur_ref_pos, bp_ref_pos, None, None), None, None, None, 'None', None, height)
                    pile_column_record.append(pile_result)
                    height += 1
                pile_result = ((cur_ref_pos, bp_ref_pos, cur_query_pos, bp_query_pos),
                               map_type,  # eg:0(M),4(S),5(H)
                               pile_read.alignment.query_sequence[pile_read.query_position],
                               cigar_str,
                               this_qname,
                               abs(pile_read.alignment.tlen),
                               new_height)
                # pile_read.alignment.is_paired,
                # pile_read.alignment.is_proper_pair,
                # pile_read.alignment.mapping_quality,
                if new_height == height:
                    pile_column_record.append(pile_result)
                if new_height < height:
                    pile_column_record[new_height] = pile_result
                    height -= 1

            # case2
            else:
                this_qname_list = [r[4] for r in pile_column_record]
                if 'None' in this_qname_list:  #
                    first_non_index = this_qname_list.index('None')
                    new_height = pile_column_record[first_non_index][-1]  #
                    pile_result = ((cur_ref_pos, bp_ref_pos, cur_query_pos, bp_query_pos),
                                   map_type,  # eg:0(M),4(S),5(H)
                                   pile_read.alignment.query_sequence[pile_read.query_position],
                                   cigar_str,
                                   this_qname,
                                   abs(pile_read.alignment.tlen),
                                   new_height)
                    pile_column_record[first_non_index] = pile_result
                    if new_height < height:
                        height -= 1
                else:
                    pile_result = ((cur_ref_pos, bp_ref_pos, cur_query_pos, bp_query_pos),
                                   map_type,  # eg:0(M),4(S),5(H)
                                   pile_read.alignment.query_sequence[pile_read.query_position],
                                   cigar_str,
                                   this_qname,
                                   abs(pile_read.alignment.tlen),
                                   height)
                    pile_column_record.append(pile_result)

        # if str(cur_ref_pos) == '55763898':
        #     print('55763898 column record = ',pile_column_record)
        pile_record.extend(pile_column_record)
        last_pile_column_record = pile_column_record
    return pile_record, qname_list


def get_depth(sam_file, chr_id, pos_l, pos_r):
    read_depth = sam_file.count_coverage(chr_id, pos_l, pos_r)
    depth = np.array(list(read_depth)).sum(axis=0)
    depth = list(depth)
    return depth

# # init image
# def init_pic(row, col, th, fig, flag):
#     if flag == '2d':
#         ax = fig.add_subplot(row, col, th)
#         ax.get_xaxis().get_major_formatter().set_useOffset(False)
#         return ax
#     elif flag == '3d':
#         ax = fig.add_subplot(row, col, th, projection='3d')
#         ax.get_xaxis().get_major_formatter().set_useOffset(False)
#         return ax
