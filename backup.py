import os
import shutil
import sys

source_dir = './models'
dst_dir = './result'
if os.path.exists(dst_dir): 
    shutil.rmtree(dst_dir)
os.mkdir(dst_dir)       

if __name__ == '__main__':
    for folder in os.listdir(source_dir):
        if 'GW' in folder:
            print('Copying from ', folder)
            dst_folder = os.path.join(dst_dir, folder)
            if not os.path.exists(dst_folder):
                os.mkdir(dst_folder)
            src_folder = os.path.join(source_dir, folder)
            png_file = os.path.join(src_folder, '*.png')
            txt_file = os.path.join(src_folder, '*.txt')
            npy_file = os.path.join(src_folder, '*.npy')
            os.system('cp -rf '+ png_file + ' ' + dst_folder)
            os.system('cp -rf '+ txt_file + ' ' + dst_folder)
#             os.system('cp -rf '+ npy_file + ' ' + dst_folder)
    
    print('finished...')
    print('file size:')
    os.system('du -sh '+ dst_dir)