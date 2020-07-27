# encoding: utf-8

def split_canset(full_can_path, full_del_path, full_ins_path, full_dup_path):
    f = open(full_can_path, 'r')
    f1 = open(full_del_path, 'w+')
    f2 = open(full_ins_path, 'w+')
    f3 = open(full_dup_path, 'w+')
    for line in f:
        # print(line)
        record = line.rstrip('\n').split(' ')
        if record[3] == 'DEL':
            f1.write(line)
        elif record[3] == 'INS':
            f2.write(line)
        elif record[3] == 'DUP':
            f3.write(line)
    f.close()
    f1.close()
    f2.close()
    f3.close()
    return


def get_negset(full_path, pos_path, neg_path):
    f1 = open(full_path, 'r')
    f2 = open(pos_path, 'r')
    resf = open(neg_path, 'w+')

    record1 = []
    for line in f1:
        record1.append(line)

    record2 = []
    for line in f2:
        record2.append(line)

    chrom_list = [str(a) for a in range(1, 23)]
    chrom_list.extend(['X', 'Y'])

    for record in record1:
        if record in record2:
            continue
        else:
            if record[0] in chrom_list:
                resf.write(record)

    f1.close()
    f2.close()
    resf.close()
    return


def get_del(full_path, del_path):
    f = open(full_path, 'r')
    f1 = open(del_path, 'w+')
    for line in f:
        record = line.rstrip('\n').split(' ')
        if record[3] == 'DEL' and int(record[2]) - int(record[1]) >= 50:
            f1.write(line)
    f.close()
    f1.close()
    return

def get_del_compare_res(full_path, del_path):
    f = open(full_path, 'r')
    f1 = open(del_path, 'w+')
    for line in f:
        record = line.rstrip('\n').split(' ')
        if record[9] == 'DEL' and int(record[8])-int(record[7])>=50:
            f1.write(line)
    f.close()
    f1.close()
    return


def count_full(full_can_path):
    f = open(full_can_path, 'r')
    types = dict()
    types['DUP'] = 0
    types['INV'] = 0
    types['BND'] = 0
    types['DEL'] = 0
    types['INS'] = 0
    for line in f:
        record = line.rstrip('\n').split('\t')
        types[record[3]] += 1
    print(types, sum(types.values()))


def rejust(bed_path, res_path):
    f = open(bed_path, 'r')
    r = open(res_path, 'w')
    for line in f:
        record = line.rstrip('\n').split(' ')
        if (int(record[2]) - 1) - int(record[1]) >= 50:
            res = record[0] + ' ' + str(int(record[1]) + 1) + ' ' + str(int(record[2]) - 1) + ' ' + record[3] + '\n'
            r.write(res)
    f.close()
    r.close()
    return


def get_long_del(pos_del_path, long_pos_del_path):
    f = open(pos_del_path, 'r')
    res_f = open(long_pos_del_path, 'w+')
    for line in f:
        record = line.rstrip('\n').split(' ')
        if record[3] == 'DEL':
            if int(record[2]) - int(record[1]) >= 50:
                res_f.write(line)
    f.close()
    res_f.close()
    return
