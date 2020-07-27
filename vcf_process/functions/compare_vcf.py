# encoding: utf-8
import vcf


def judge(x1, y1, x2, y2):
    left = max(x1, x2)
    right = min(y1, y2)
    common_len = float(right - left)
    overlap_rate1 = common_len / float(y1 - x1)
    overlap_rate2 = common_len / float(y2 - x2)

    if common_len <= 0:
        return False, 0, 0
    # if common_len > 10:
    if overlap_rate1 >= 0.9 and overlap_rate2 >= 0.9:
        return True, overlap_rate1, overlap_rate2
    # else:
    #     if overlap_rate1 >= 0.4 and overlap_rate2 >= 0.4:
    #         return True, overlap_rate1, overlap_rate2

    return False, overlap_rate1, overlap_rate2


def compare_vcf(truth_vcf_path, can_vcf_path, compare_res_path, match_res_path, check_res_path):
    orig_f = open(truth_vcf_path, 'r')
    can_f = open(can_vcf_path, 'r')

    compare_res_f = open(compare_res_path, 'w')
    match_res_f = open(match_res_path, 'w')
    check_res_f = open(check_res_path, 'w')

    record1 = []
    record2 = []
    chrom_list = [str(a) for a in range(1, 23)]
    chrom_list.extend(['X', 'Y'])

    for chrom_id in chrom_list:
        break_chr = '\t'

        orig_line = orig_f.readline().rstrip('\n')
        if orig_line and len(orig_line.split('\t')) == 1:
            break_chr = ' '
        while orig_line:
            orig_record = orig_line.split(break_chr)
            if orig_record[0] == chrom_id:
                record1.append(orig_record)
            orig_line = orig_f.readline().rstrip('\n')

        can_line = can_f.readline().rstrip('\n')
        if can_line and len(can_line.split('\t')) == 1:
            break_chr = ' '
        while can_line:
            can_record = can_line.split(break_chr)
            if can_record[0] == chrom_id:
                record2.append(can_record)
            can_line = can_f.readline().rstrip('\n')

        if record1 and record2:
            orig_f.close()
            can_f.close()

            record1 = sorted(record1, key=lambda x: float(x[1]))
            record2 = sorted(record2, key=lambda x: float(x[1]))

            to_check(record1, record2, compare_res_f, match_res_f, check_res_f)

            record1 = []
            record2 = []

            orig_f = open(truth_vcf_path, 'r')
            can_f = open(can_vcf_path, 'r')

    orig_f.close()
    can_f.close()
    compare_res_f.close()
    match_res_f.close()
    check_res_f.close()

    return


def to_check(record1, record2, compare_res_f, res_f, check_res_f):
    for j in range(len(record2)):
        print("===chr", record2[j][0], "===processing ", str(j + 1) + '/' + str(len(record2)), "===", end='\r')
        record = record2[j]
        left = float(record[1])
        right = float(record[2])
        length = right - left + 1
        if length > 10:
            stretch = 0.5  #0.12  # 1/9
        else:
            stretch = 1.0
        min_left, min_right = left - stretch * length, right - stretch * length
        max_left, max_right = left + stretch * length, right + stretch * length
        best_check_line = ""
        best_overlap_rate = -1
        for i in range(len(record1)):
            l, r = float(record1[i][1]), float(record1[i][2])
            # print("min_range:", (min_left, max_left), " right_range:", (min_right, max_right))
            if l > max_left:
                break
            if (min_left <= l <= max_left) and (min_right <= r <= max_right):
                # print("in range:", (l, r))
                is_match, overlap_rate1, overlap_rate2 = judge(l, r, left, right)
                if is_match:
                    compare_res_line = ' '.join(record1[i]) + ' ' + ' '.join(
                        record) + ' ' + str(overlap_rate1) + ' ' + str(overlap_rate2) + '\n'
                    # print("[ MATCH! " + compare_res_line + " ]")
                    compare_res_f.write(compare_res_line)

                    res_line = ' '.join(record) + '\n'
                    res_f.write(res_line)
                    break
                else:
                    if max(overlap_rate1, overlap_rate2) > best_overlap_rate:
                        best_overlap_rate = max(overlap_rate1, overlap_rate2)
                        best_check_line = ' '.join(record1[i]) + ' ' + ' '.join(
                            record) + ' ' + str(overlap_rate1) + ' ' + str(overlap_rate2) + '\n'
        check_res_f.write(best_check_line)
    return


def abandoned_to_check(record1, record2, i, j, res):
    if i >= len(record1) or j >= len(record2):
        return
    while float(record1[i][2]) < float(record2[j][1]):
        # print("[Candidate goes far than Orig!] -> Orig+1")
        i += 1
    while float(record1[i][1]) - float(record2[j][1]) > 100000 or float(record1[i][1]) > float(record2[j][2]):
        # print("[Candidate lay back than Orig!] -> Candidate+1")
        j += 1
    if judge(float(record1[i][1]), float(record1[i][2]), float(record2[j][1]), float(record2[j][2])):
        # and ((record1[i][4] == record2[j][3] == 'DUP') or (
        #         record1[i][4] == 'SIMPLEDEL' and record2[j][3] == 'DEL') or (
        #              record1[i][4] == 'SIMPLEINS' and record2[j][3] == 'INS')) \
        if (i, j) in res:
            i += 1
            j += 1
            return
        print("[MATCH] -> Candidate+1")
        res.add((i, j))
        compare_res_line = ' '.join(record1[i]) + ' ' + ' '.join(record2[j])
        print(compare_res_line)
        print("res=", res)
        i += 1
        j += 1
    else:
        # print("[NOT MATCH! ")
        abandoned_to_check(record1, record2, i + 1, j, res)
        abandoned_to_check(record1, record2, i, j + 1, res)

