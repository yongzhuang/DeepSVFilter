# DeepSVFilter
## Introduction 
DeepSVFilter is a deep learning based tool designed to filter false positive structural variants (SVs) obtained by any computational approaches from short read whole genome sequencing data. It can be used as either a stand-alone tool to filter SVs or coupled with commonly used SV detection tool (Delly, Lumpy, Manta et al.) to improve specificity.

## Installation
### Dependencies
tensorflow 1.15.0, matplotlib==3.1.0, numpy<2.0,>=1.16.0, opencv-python==3.1.0.4, Pillow==7.2.0, pysam==0.15.4, scikit-learn==0.19.2, scipy
### Install from github (requires Python 3.6.* or newer)
1. https://github.com/yongzhuang/DeepSVFilter.git
2. cd DeepSVFilter
3. pip install .

## Running
usage: DeepSVFilter [OPTIONS]  

**1. preprocess**
      This option is used to generate SV images for candidate SVs.

usage: DeepSVFilter preprocess [OPTIONS]  

--sv_type	<STR>	SV type (DEL or DUP) (required)  
	
--bam_path		<FILE>	BAM file (required)  
	
--bed_path		<FILE>	SV BED file (required)  
	
--patch_size	<INT>	image patch size (224 or 299) (required)  
	
--output_imgs_dir	<DIR>	output image folder (required  
	
--mean_insert_size	<INT>	mean of the insert size (optional)  
	
--sd_insert_size	<INT>	standard deviation of the insert size (optional)  

**2. augmentate**
      This option is used to do data augmentation for typical SVs.  

usage: DeepSVFilter augmentate [OPTIONS]   

--output_imgs_dir	<DIR>	output image folder (required)  
--image_path_file	<FILE>	input typical true or false image path file (required)  
--patch_size		<INT>	image patch size (224 or 299) (required)  

**3. train**
      This option is used to train a convolutional neural network (CNN) based SV classification model.  

usage: DeepSVFilter train [OPTIONS]   

--sv_type	SV type (DEL or DUP) (required)  
--checkpoint_dir	<DIR>	checkpoint folder (required)  
--pos_train_file	<FILE>	path file of positive SV images used for training (required)  
--neg_train_file	<FILE>	path file of negative SV images used for training (required)  
--pos_eval_file		<FILE>	path file of positive SV images used for evaluation (required)  
--neg_eval_file		<FILE>	path file of negative SV images used for evaluation (required)  
--eval_result_dir	<DIR>	validation result (required)  
--summary_dir		<DIR>	tensorboard summary (required)  
--use_gpu	gpu flag, 1 for GPU and 0 for CPU (optional, default 0)  
--gpu_idx	GPU idx (optional, default 0)  
--gpu_mem	gpu memory usage (0 to 1) (optional, default 0.5)  
--model				<STR>	M1(for MobileNet_v1) or M2_1.0(for MobileNet_v2_1.0) or M2_1.4(for MobileNet_v2_1.4) or NAS(for NASNet_A_Mobile) or PNAS(for PNASNet_5_Mobile) or IR_v2(for Inception_ResNet_v2) (optional, default M1)  
--epoch 			<INT>	number of total epoches (optional, default 13)  
--batch_size		<INT>	number of samples in one batch (optinal, default 16)  
--start_lr			<INT>	initial learning rate for adam (optional, default 0.001)  
--eval_every_epoch	<INT>	evaluating and saving checkpoints every # epoch (optional, default 1)  
--num_cores			<INT>	maximum number of CPU cores (optional, default: use all cpu cores)   

**4. predict**
      This option is used to make predictions for candidate SVs, and the SVs with scores less than the specified threshold (default 0.5) are filtered out.  

usage: DeepSVFilter predict [OPTIONS]  

--sv_type	SV type (DEL or DUP) (required)  
--checkpoint_dir	<DIR>	checkpoint folder (required)  
--test_file			<FILE>	SV image path file (required)  
--test_result_dir	<DIR>	SV filtering results (required)  
--use_gpu	gpu flag, 1 for GPU and 0 for CPU (optional, default 0)  
--gpu_idx	GPU idx (optional, default 0)  
--gpu_mem	gpu memory usage (0 to 1) (optional, default 0.5)  
--model				<STR>	M1(for MobileNet_v1) or M2_1.0(for MobileNet_v2_1.0) or M2_1.4(for MobileNet_v2_1.4) or NAS(for NASNet_A_Mobile) or PNAS(for PNASNet_5_Mobile) or IR_v2(for Inception_ResNet_v2) (optional, default M1)   

**5. vcf2bed**
	This option is used to used to convert SV vcf file to bed file.  

usage: vcf2bed [OPTIONS]  

--sv_type	<STR>	SV Type (DEL or DUP) (required)  
--vcf_file	<FILE>	vcf file (required)  
--bed_file	<FILE>	bed file (required)  
--tool_name <STR>	delly,manta,lumpy or giab (required)  
--exclude	<FILE>	exclude bed file (optional, default NULL)  
--length	<INT>	SV length (optional, default 100)  

**6. extract_typical_SV**
	This optional is used to extract typical SVs defined in the manuscript.  

usage: extract_typical_SV [OPTIONS]  

--sv_type	<STR>	SV Type (DEL or DUP) (required)  
--vcf_file	<FILE>	vcf file (required)  
--bed_file	<FILE>	bed file (required)  
--ground_truth_file <FILE>	ground truth vcf file (required)  
--tool_name <STR>	delly,manta,lumpy or giab (required)  
--exclude	<FILE>	exclude bed file (optional, default NULL)  
--length	<INT>	SV length (optional, default 100)  

## Input and Output

1. The 'preprocess' command will take a SV bed file and output a SV image directory which contains  

1) image dir: storing all SV images  

2) SV image path file: storing the paths of all SV images  

2. The 'augmentate' command will take a SV image path file and also output a SV image directory after data augmentation.  

3. The 'train' command will take four SV image path files and output the trained model in the checkpoint directory.  

4. The 'predict' command will take the SV image path file got by the 'preprocess' command and output the filtering result  as follows:  
   
   Column 1: chromosome  
   Column 2: start  
   Column 3: end  
   Coumnn 4: score   

## Example

**1. run preprocess.sh to get candidate SV images**  

 DeepSVFilter preprocess \  
        --sv_type=DEL \  
        --patch_size=224 \  
        --bam_path=./data/example.bam \  
        --bed_path=./data/example.bed \  
        --output_imgs_dir=./result/images  

**2. run predict.sh to make predictions for candidate SVs**  

 DeepSVFilter predict \  
 	--sv_type DEL \  
 	--test_file ./result/images/IMG_PATH.txt \  
 	--checkpoint_dir ./checkpoint \  
 	--test_result_dir ./result/filteredSVs  

## Running Time and Memory Requirements

1. The training time is less than 12 hours for a typical size of training set (about 10,000 positive and negative examples).  

2. The predicting time is less than 1 hour for a typical human genome (about 5000 candidate SVs).  

3. All analysis in the manuscript were run on a 384GB memory server.  

## Contact 
   yongzhuang.liu@hit.edu.cn
   
## License
[MIT](https://github.com/yongzhuang/TumorCNV/blob/master/LICENSE)
