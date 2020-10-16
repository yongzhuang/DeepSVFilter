import os
from PIL import Image

def data_augmentation(image_path_file,patch_size,output_imgs_dir):
    
    if not os.path.exists(output_imgs_dir):
        os.makedirs(output_imgs_dir)
    image_output_dir = os.path.join(output_imgs_dir, "image")
    image_output_dir = os.path.abspath(image_output_dir)
    if not os.path.exists(image_output_dir):
        os.makedirs(image_output_dir)
    
    file_path = os.path.join(output_imgs_dir, "IMG_PATH.txt")

    image_path_file_reader = open(image_path_file, 'r')
    image_path_list = [line.rstrip('\n') for line in image_path_file_reader]
    
    f = open(file_path, 'w')
    for image_path in image_path_list:
        im = Image.open(image_path)
        left_image=im.crop((0,0,patch_size,patch_size//2))
        right_image=im.crop((0,patch_size//2+1,patch_size, patch_size))
        vertical_im = Image.new('RGB', (patch_size, patch_size), (0, 255, 255))
        vertical_im.paste(right_image.transpose(Image.FLIP_LEFT_RIGHT), (0, 0))
        vertical_im.paste(left_image.transpose(Image.FLIP_LEFT_RIGHT), (0, patch_size//2+1))
        
        raw_file_name=os.path.basename(image_path)
        new_file_name=raw_file_name[0:-4]+'_DA'+raw_file_name[-4:]
 
        save_path = os.path.join(image_output_dir, new_file_name)
        vertical_im.save(save_path, "PNG")
        f.write(str(save_path) + '\n')
    f.close()

if __name__ == '__main__':
    image_path_file='/data/yzliu/DeepSVFilter3/augmentation/input.txt'
    output_imgs_dir='./test'
    data_augmentation(image_path_file,224,output_imgs_dir)

