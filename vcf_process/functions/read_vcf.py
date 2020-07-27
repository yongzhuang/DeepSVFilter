# encoding: utf-8
import vcf


def get_vcf_svtype(vcf_path):
    res = set()
    vcf_reader = vcf.Reader(filename=vcf_path)
    for record in vcf_reader:
        if record.var_type == 'sv':
            res.add(record.INFO['SVTYPE'])  # , record.var_subtype)
    return list(res)


def read_vcf(vcf_path, res_path):
    f = open(res_path, 'w+')
    # f.write("CHROM  POS-1    END    INFO['SVTYPE'] INFO['REPTYPE']\n")
    vcf_reader = vcf.Reader(filename=vcf_path)
    line = ""
    for record in vcf_reader:
        if record.var_type == 'sv':
            line = str(record.CHROM) + ' ' + str(record.POS - 1) + ' ' + str(record.INFO['END']) + ' ' + str(
                record.INFO['SVTYPE']) + '  ' + str(record.INFO['REPTYPE']) + '\n'
            f.write(line)
    f.close()
    return


def read_restrict_vcf(vcf_path, res_path):
    f = open(res_path, 'w+')
    # f.write("CHROM  POS-1    END    INFO['SVTYPE'] INFO['REPTYPE']\n")
    vcf_reader = vcf.Reader(filename=vcf_path)
    line = ""
    for record in vcf_reader:
        if record.INFO['REPTYPE'] in ['SIMPLEDEL', 'SIMPLEINS']:
            line = 'chr' + str(record.CHROM) + ' ' + str(record.POS - 1) + ' ' + str(record.INFO['END']) + ' ' + str(
                record.INFO['SVTYPE']) + '  ' + str(record.INFO['REPTYPE']) + '\n'
            f.write(line)
    f.close()
    return


def read_delly_vcf(delly_vcf_path, res_path):
    f = open(res_path, 'w+')
    vcf_reader = vcf.Reader(filename=delly_vcf_path)
    line = ""
    for record in vcf_reader:
        # if record.INFO['SVTYPE'] in ['DEL', 'INS', 'DUP', 'INV']:
        info = None
        if 'PRECISE' in record.INFO.keys():
            info = 'PRECISE'
        elif 'IMPRECISE' in record.INFO.keys():
            info = 'IMPRECISE'
        line = str(record.CHROM) + ' ' + str(record.POS - 1) + ' ' + str(record.INFO['END'] - 1) + ' ' + str(
            record.INFO['SVTYPE']) + ' ' + str(record.FILTER) + ' ' + info + '\n'
        f.write(line)
    f.close()
    return


def read_manta_vcf(manta_vcf_path, res_path):
    f = open(res_path, 'w+')
    vcf_reader = vcf.Reader(filename=manta_vcf_path)
    line = ""
    for record in vcf_reader:
        #TODO: manta这里这么写不对，因为好多没有PRECISE和IMPRECISE字段，就都成PRECISE了
        if 'IMPRECISE' in record.INFO.keys():
            info = 'IMPRECISE'
        else:
            info = 'PRECISE'

        end = record.POS + 1
        if 'END' in record.INFO.keys():
            end = record.INFO['END']
        line = str(record.CHROM) + ' ' + str(record.POS - 1) + ' ' + str(end) + ' ' + str(
            record.INFO['SVTYPE']) + ' ' + str(record.FILTER) + ' ' + info + '\n'
        f.write(line)
    f.close()
    return


def read_lumpy_vcf(lumpy_vcf_path, res_path):
    f = open(res_path, 'w+')
    vcf_reader = vcf.Reader(filename=lumpy_vcf_path)
    line = ""
    for record in vcf_reader:
        if 'IMPRECISE' in record.INFO.keys():
            info = 'IMPRECISE'
        else:
            info = 'PRECISE'

        end = record.POS + 1
        if 'END' in record.INFO.keys():
            end = record.INFO['END']
        line = str(record.CHROM) + ' ' + str(record.POS - 1) + ' ' + str(end) + ' ' + str(
            record.INFO['SVTYPE']) + ' ' + str(record.FILTER) + ' ' + info + '\n'
        f.write(line)
    f.close()
    return


def read_delly_vcf2(delly_vcf_path, res_path):
    vcf_f = open(delly_vcf_path, 'r')
    bed_f = open(res_path, 'w+')
    # vcf_reader = vcf.Reader(filename=delly_vcf_path)
    res = ""
    while True:
        line = vcf_f.readline().rstrip('\n')
        if line:
            if line[0] == '#':
                continue
            else:
                record = line.split('\t')
                res = str(record[0]) + ' ' \
                      + str(int(record[1]) - 1) + ' ' \
                      + record[7].split(';')[3].split('=')[1] + ' ' \
                      + record[7].split(';')[1].split('=')[1] + '\n'
                # print("res=", res)
                bed_f.write(res)
        else:
            break
    bed_f.close()
    return


def write_bed(vcf_path, res_path):
    f = open(res_path, 'w+')
    vcf_reader = vcf.Reader(filename=vcf_path)
    line = ""

    for record in vcf_reader:
        if record.var_type == 'sv':
            line = str(record.CHROM) + '  ' + str(record.POS - 1) + '   ' + str(record.INFO['END']) + '\n'
            f.write(line)
    f.close()
    return


def get_vcf_length(vcf_path):
    min_length = float('inf')
    max_length = -1
    f = open(vcf_path, 'r')

    line = f.readline().rstrip('\n')
    while line:
        record = line.split('\t')
        len = float(record[2]) - float(record[1]) + 1
        max_length = max(len, max_length)
        min_length = min(len, min_length)
        line = f.readline().rstrip('\n')
    print("max_length=", max_length, " min_length=", min_length)
    f.close()
    return


def filt_delly_bed(bed_path, res_path):
    max_length = 3306654
    f1 = open(bed_path, 'r')
    f2 = open(res_path, 'w')

    line = f1.readline().rstrip('\n')
    while line:
        record = line.split(' ')
        len = float(record[2]) - float(record[1]) + 1
        if len <= max_length:
            line += '\n'
            f2.write(line)
        line = f1.readline().rstrip('\n')
    f1.close()
    f2.close()
    return


def delly_bed_adjust(delly_bed_path, adjusted_delly_bed_path):
    f = open(delly_bed_path, 'r')
    res_f = open(adjusted_delly_bed_path, 'w')
    for line in f:
        record = line.rstrip('\n').split(' ')
        res = record[0] + ' ' + record[1] + ' ' + str(int(record[2]) - 1) + ' ' + record[3] + '\n'
        res_f.write(res)
    f.close()
    res_f.close()
    return

