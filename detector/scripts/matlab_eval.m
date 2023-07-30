root = [pwd '/../..'];
src_path = [root '/data/VOCdevkit2007'];
com_id = 'comp4';
img_set = 'test';
output_dir = [root '/detector/output'];
script_path = [root '/detector/lib/datasets/VOCdevkit-matlab-wrapper'];
cd(script_path)
voc_eval(src_path,com_id,img_set,output_dir);