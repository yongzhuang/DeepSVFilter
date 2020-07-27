# DeepSVFilter

## Requirements
1. Ubuntu 16.04.6
2. Python3.6.4, Tensorflow1.12.0, Keras2.2.4
3. Numpy1.14.0, opencv-python3.1.0, PIL7.1.2, Scikit-Learn0.19.2, Matplotlib3.1.0
4. Pysam0.15.4, Samtools, PyVCF0.6.8
 
 ## Quick Start : Test on sample images
 - *Using testing data list in ```data/test/```, test ROC curve figure can be find in ```test_results/```.*<br>
```shell
python main.py --phase=test 
```

## Testing on Your Own Dataset
#### 1.Transform Your Bed File to Images
```shell
python main.py \  
    --use_gpu=0 \                           # use gpu or not
    --phase=preprocess \
    --sv_type=DEL \
    --patch_size=224 \                      #224 or 299
    --bam_path=/path/to/your/bam/file.bam \                # bam filepath
    --p_bed_path=/path/to/your/positive/bed/P.bed \         # positive bed directory
    --n_bed_path=/path/to/your/negative/bed/N.bed \         # negative bed directory
    --output_imgs_dir=imgs/
```
#### 2.Generating Image Filepath as testing data
 ```shell
python utils/file_travel.py \ 
    --img_p_dir=/path/to/your/positive/img/dir/P/ \         # positive images directory
    --img_n_dir=/path/to/your/negative/img/dir/N/ \         # negative images directory 
    --output_dir=/path/to/your/output/data/dir/ \           # output image list directory  
    --usage=test                                           # are those training images or testing images
```
#### 3.Start Testing
 ```shell
python main.py --phase=test --test_dir=data/test/
    --checkpoint_dir=/path/to/your/negative/bed/dir/ \    
    --summary_dir=/path/to/your/negative/bed/dir/ \    
```

## Training on Your Own Dataset
#### 1.Transform Your Bed File to Images
```shell
python main.py \  
    --use_gpu=0 \                           # use gpu or not
    --phase=preprocess \
    --sv_type=DEL \
    --patch_size=224 \                      #224 or 299
    --bam_path=/path/to/your/bam/file.bam \                # bam filepath
    --p_bed_path=/path/to/your/positive/bed/P.bed \         # positive bed directory
    --n_bed_path=/path/to/your/negative/bed/N.bed \         # negative bed directory
    --output_imgs_dir=imgs/
```
#### 2.Generating Image Filepath List as training data
 ```shell
python utils/file_travel.py \ 
    --img_p_dir=/path/to/your/positive/img/dir/P/ \         # positive images directory
    --img_n_dir=/path/to/your/negative/img/dir/N/ \         # negative images directory 
    --output_dir=/path/to/your/output/data/dir/ \           # output image list directory  
    --usage=train                                           # are those training images or testing images
    --train_percent=0.9                                     # use how many percent images for training(only use if usage=train)
```
#### 3.Start Training
 ```shell
python main.py 
    --use_gpu=0 \                           # use gpu or not
    --phase=train \
    --sv_type=DEL \
    --model=M1
    --epoch=13
    --batch_size=16
    --start_lr=0.001
    --eval_every_epoch=1
    --checkpoint_dir=/path/to/your/checkpoint/dir/ \  
    --train_dir=/path/to/your/data/train/ 
    --eval_dir=/path/to/your/data/eval/ 
    --eval_results_dir=/path/to/your/data/eval/results/
    --summary_dir=/path/to/your/summary/dir/ \    
