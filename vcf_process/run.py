# coding=utf-8

from __future__ import print_function
import sys
import os
from glob import glob

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import argparse
from vcf_process.functions.compare_vcf import *
from vcf_process.functions.read_vcf import *
from vcf_process.functions.split_canset import *



parser = argparse.ArgumentParser(description='')

parser.add_argument('--sv_type', dest='sv_type', default='DEL', help='DEL, INS or DUP')

# preprocess-vcf stage
parser.add_argument('--truth_vcf_path', dest='truth_vcf_path',
                    default='../vcf/HG002_SVs_Tier1_v0.6.vcf',
                    help='filepath for truth_set vcf file', required=True)
parser.add_argument('--candidate_vcf_dir', dest='candidate_vcf_dir',
                    default='../vcf/candidate/',
                    help='directory for vcf file to process(can deal delly/manta/lumpy vcf only)')
parser.add_argument('--output_bed_dir', dest='output_bed_dir',
                    default='../bed/', help='directory for output bed')

args = parser.parse_args()

def vcf_routine(truth_bed_path, can_vcf_path, bed_output_dir):
    # Part1
    output_bed_path = (bed_output_dir + can_vcf_path.split('/')[-1].split('.vcf')[0] + '.bed')
    if "delly" in output_bed_path:
        read_delly_vcf(can_vcf_path, output_bed_path)
    elif "manta" in output_bed_path:
        read_manta_vcf(can_vcf_path, output_bed_path)
    elif "lumpy" in output_bed_path:
        read_lumpy_vcf(can_vcf_path, output_bed_path)
    else:
        print("Unknown Type")
        exit(1)
    print("[*] Part1: " + can_vcf_path.split('/')[-1].split('.vcf')[0] + "'s bed save to: ", output_bed_path)

    # Part2
    output_match_bed_path = (bed_output_dir + 'positive/' + can_vcf_path.split('/')[-1].split('.vcf')[
        0] + '.match_res.bed')
    output_compare_bed_path = (bed_output_dir + 'compare/' + can_vcf_path.split('/')[-1].split('.vcf')[
        0] + '.compare_res.bed')
    output_close_bed_path = (bed_output_dir + 'close/' + can_vcf_path.split('/')[-1].split('.vcf')[
        0] + '.close_res.bed')
    if not os.path.exists(bed_output_dir + 'positive/'):
        os.makedirs(bed_output_dir + 'positive/')
    if not os.path.exists(bed_output_dir + 'compare/'):
        os.makedirs(bed_output_dir + 'compare/')
    if not os.path.exists(bed_output_dir + 'close/'):
        os.makedirs(bed_output_dir + 'close/')
    compare_vcf(truth_bed_path, output_bed_path, output_compare_bed_path,
                output_match_bed_path, output_close_bed_path)
    print("[*] Part2(1/3): " + can_vcf_path.split('/')[-1].split('.vcf')[0] + "'s match bed save to: ", output_match_bed_path)
    print("[*] Part2(2/3): " + can_vcf_path.split('/')[-1].split('.vcf')[0] + "'s compare results save to: ", output_compare_bed_path)
    print("[*] Part2(3/3): " + can_vcf_path.split('/')[-1].split('.vcf')[0] + "'s close results save to: ", output_close_bed_path)

    # Part3
    output_wrong_bed_path = (bed_output_dir + 'negative/' + can_vcf_path.split('/')[-1].split('.vcf')[
        0] + '.wrong_res.bed')
    if not os.path.exists(bed_output_dir + 'negative/'):
        os.makedirs(bed_output_dir + 'negative/')
    get_negset(output_bed_path, output_match_bed_path, output_wrong_bed_path)
    print("[*] Part3: " + can_vcf_path.split('/')[-1].split('.vcf')[0] + "'s negative bed save to: ", output_wrong_bed_path)
    print("[*] End of processing "+can_vcf_path.split('/')[-1].split('.vcf')[0])
    return

if __name__ == '__main__':
    delly_vcf_path = glob(args.candidate_vcf_dir + 'HG002*delly*.vcf')[0]
    manta_vcf_path = glob(args.candidate_vcf_dir + 'HG002*manta*.vcf')[0]
    lumpy_vcf_path = glob(args.candidate_vcf_dir + 'HG002*lumpy*.vcf')[0]
    # vcf_main(args.truth_vcf_path, delly_vcf_path, lumpy_vcf_path, manta_vcf_path, args.output_bed_dir)
    print("[*] Processing Truth VCF to BED file")
    output_truth_bed_path = args.output_bed_dir + args.truth_vcf_path.split('/')[-1].split('.vcf')[0] + '.bed'
    read_vcf(args.truth_vcf_path, output_truth_bed_path)
    print("[*] BED file have been saved to:", output_truth_bed_path)

    print("[*] Processing Candidate VCFs to BED File")
    if delly_vcf_path:
        print("[*] Processing Delly VCF")
        vcf_routine(output_truth_bed_path, delly_vcf_path, args.output_bed_dir)
    if lumpy_vcf_path:
        print("[*] Processing Lumpy VCF")
        vcf_routine(output_truth_bed_path, lumpy_vcf_path, args.output_bed_dir)
    if manta_vcf_path:
        print("[*] Processing Manta VCF")
        vcf_routine(output_truth_bed_path, manta_vcf_path, args.output_bed_dir)
    print("[*] Processing Results have been saved to:", args.output_bed_dir)
