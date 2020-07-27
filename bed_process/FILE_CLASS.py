# encoding: utf-8
import pysam


class File:
    def get_bam_file(self, bam_path):
        bam_file = pysam.AlignmentFile(bam_path, "rb")
        if not bam_file:
            print("bam_file is empty")
            return
        return bam_file

    def get_bed_file(self, bed_path):
        f = open(bed_path, 'r')
        if not f:
            print('[!] BED Path: [' + bed_path + '] is Empty')
            exit(1)
            return

        res = []
        for line in f:
            line = line.rstrip('\n')
            if len(line.split('\t')) == 1:
                record = line.split(' ')
            else:
                record = line.split('\t')
            new_record = []
            for r in record:
                if len(r) < 1:
                    continue
                else:
                    new_record.append(r)
            record = new_record
            res.append((str(record[0]), int(record[1]) + 1, int(record[2]), record[-1]))
        return res
