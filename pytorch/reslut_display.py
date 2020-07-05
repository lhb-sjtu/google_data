import numpy as np
import os
import cv2
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pycocotools import mask as maskUtils
from test_data_load import *
import argparse
import sys
from PIL import Image

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg



parser = argparse.ArgumentParser(description='BTS PyTorch implementation.', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--mode',                      type=str,   help='train or test', default='train')
parser.add_argument('--model_name',                type=str,   help='model name', default='bts_eigen_v2')
parser.add_argument('--encoder',                   type=str,   help='type of encoder, desenet121_bts, densenet161_bts, '
                                                                    'resnet101_bts, resnet50_bts, resnext50_bts or resnext101_bts',
                                                               default='densenet161_bts')
# Dataset
parser.add_argument('--dataset',                   type=str,   help='dataset to train on, kitti or nyu', default='nyu')
parser.add_argument('--data_path',                 type=str,   help='path to the data', required=True)
parser.add_argument('--gt_path',                   type=str,   help='path to the groundtruth data', required=True)
parser.add_argument('--filenames_file',            type=str,   help='path to the filenames text file')
parser.add_argument('--input_height',              type=int,   help='input height', default=480)
parser.add_argument('--input_width',               type=int,   help='input width',  default=640)
parser.add_argument('--max_depth',                 type=float, help='maximum depth in estimation', default=10)

# Log and save
parser.add_argument('--log_directory',             type=str,   help='directory to save checkpoints and summaries', default='')
parser.add_argument('--checkpoint_path',           type=str,   help='path to a checkpoint to load', default='')
parser.add_argument('--log_freq',                  type=int,   help='Logging frequency in global steps', default=100)
parser.add_argument('--save_freq',                 type=int,   help='Checkpoint saving frequency in global steps', default=500)

# Training
parser.add_argument('--fix_first_conv_blocks',                 help='if set, will fix the first two conv blocks', action='store_true')
parser.add_argument('--fix_first_conv_block',                  help='if set, will fix the first conv block', action='store_true')
parser.add_argument('--bn_no_track_stats',                     help='if set, will not track running stats in batch norm layers', action='store_true')
parser.add_argument('--weight_decay',              type=float, help='weight decay factor for optimization', default=1e-2)
parser.add_argument('--bts_size',                  type=int,   help='initial num_filters in bts', default=512)
parser.add_argument('--retrain',                               help='if used with checkpoint_path, will restart training from step zero', action='store_true')
parser.add_argument('--adam_eps',                  type=float, help='epsilon in Adam optimizer', default=1e-6)
parser.add_argument('--batch_size',                type=int,   help='batch size', default=4)
parser.add_argument('--num_epochs',                type=int,   help='number of epochs', default=50)
parser.add_argument('--learning_rate',             type=float, help='initial learning rate', default=1e-4)
parser.add_argument('--end_learning_rate',         type=float, help='end learning rate', default=-1)
parser.add_argument('--variance_focus',            type=float, help='lambda in paper: [0, 1], higher value more focus on minimizing variance of error', default=0.85)

# Preprocessing
parser.add_argument('--do_random_rotate',                      help='if set, will perform random rotation for augmentation', action='store_true')
parser.add_argument('--degree',                    type=float, help='random rotation maximum degree', default=2.5)
parser.add_argument('--do_kb_crop',                            help='if set, crop input images as kitti benchmark images', action='store_true')
parser.add_argument('--use_right',                             help='if set, will randomly use right images when train on KITTI', action='store_true')

# Multi-gpu training
parser.add_argument('--num_threads',               type=int,   help='number of threads to use for data loading', default=1)
parser.add_argument('--world_size',                type=int,   help='number of nodes for distributed training', default=1)
parser.add_argument('--rank',                      type=int,   help='node rank for distributed training', default=0)
parser.add_argument('--dist_url',                  type=str,   help='url used to set up distributed training', default='tcp://127.0.0.1:1234')
parser.add_argument('--dist_backend',              type=str,   help='distributed backend', default='nccl')
parser.add_argument('--gpu',                       type=int,   help='GPU id to use.', default=None)
parser.add_argument('--multiprocessing_distributed',           help='Use multi-processing distributed training to launch '
                                                                    'N processes per node, which has N GPUs. This is the '
                                                                    'fastest way to use PyTorch for either single node or '
                                                                    'multi node data parallel training', action='store_true',)
# Online eval
parser.add_argument('--do_online_eval',                        help='if set, perform online eval in every eval_freq steps', action='store_true')
parser.add_argument('--data_path_eval',            type=str,   help='path to the data for online evaluation', required=False)
parser.add_argument('--gt_path_eval',              type=str,   help='path to the groundtruth data for online evaluation', required=False)
parser.add_argument('--filenames_file_eval',       type=str,   help='path to the filenames text file for online evaluation', required=False)
parser.add_argument('--min_depth_eval',            type=float, help='minimum depth for evaluation', default=1e-3)
parser.add_argument('--max_depth_eval',            type=float, help='maximum depth for evaluation', default=80)
parser.add_argument('--eigen_crop',                            help='if set, crops according to Eigen NIPS14', action='store_true')
parser.add_argument('--garg_crop',                             help='if set, crops according to Garg  ECCV16', action='store_true')
parser.add_argument('--eval_freq',                 type=int,   help='Online evaluation frequency in global steps', default=500)
parser.add_argument('--eval_summary_directory',    type=str,   help='output directory for eval summary,'
                                                                    'if empty outputs to checkpoint folder', default='')

if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()


def default(o):
    if isinstance(o, np.int64): return int(o)
    if isinstance(o, bytes): return str(o,encoding='utf-8')
    raise TypeError


from bts import *

model = BtsModel(args)

if args.checkpoint_path != '':
    if os.path.isfile(args.checkpoint_path):
        print("Loading checkpoint '{}'".format(args.checkpoint_path))
        if args.gpu is None:
            checkpoint = torch.load(args.checkpoint_path)
        else:
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(args.checkpoint_path, map_location=loc)
        global_step = checkpoint['global_step']
        model.load_state_dict(checkpoint['model'])
        try:
            best_eval_measures_higher_better = checkpoint['best_eval_measures_higher_better'].cpu()
            best_eval_measures_lower_better = checkpoint['best_eval_measures_lower_better'].cpu()
            best_eval_steps = checkpoint['best_eval_steps']
        except KeyError:
            print("Could not load values for online evaluation")

        print("Loaded checkpoint '{}' (global_step {})".format(args.checkpoint_path, checkpoint['global_step']))
    else:
        print("No checkpoint found at '{}'".format(args.checkpoint_path))
model.eval()

def random_crop( img, depth, height, width):
    assert img.shape[0] >= height
    assert img.shape[1] >= width
    assert img.shape[0] == depth.shape[0]
    assert img.shape[1] == depth.shape[1]
    x = random.randint(0, img.shape[1] - width)
    y = random.randint(0, img.shape[0] - height)
    img = img[y:y + height, x:x + width, :]
    depth = depth[y:y + height, x:x + width, :]
    return img, depth




data_path = "/home/lei/PycharmProjects/test-simulation/timber_cube_eval"
data_name = "00000.png"


num = 0
data_all = []
# 深度图片的名字 00001.png
image_path = os.path.join(data_path, "Images", data_name)
depth_path = os.path.join(data_path, "Depths", data_name)
image = Image.open(image_path)
depth_gt = Image.open(depth_path)
depth_gt = depth_gt.crop((43, 45, 608, 472))
image = image.crop((43, 45, 608, 472))
image = np.asarray(image, dtype=np.float32) / 255.0
depth_gt = np.asarray(depth_gt, dtype=np.float32)
depth_gt = np.expand_dims(depth_gt, axis=2)
depth_gt = depth_gt / 1000.0
image, depth_gt = random_crop(image, depth_gt, 416, 544)
image = np.array([np.transpose(image, (2, 0, 1))])
depth_gt = np.array([np.transpose(depth_gt, (2, 0, 1))])
image = torch.tensor(image)
depth_gt = torch.tensor(depth_gt)

_, _, _, _, pred_depth = model(image, 500)
pred_depth = pred_depth.detach().numpy().squeeze()
depth = pred_depth
# 通道拆分 将深度信息分出来
# depth[depth > 500] = 0
# print(depth.shape)
points = []

for j in range(0, depth.shape[0] - 1, 2):
    for k in range(0, depth.shape[1] - 1, 2):
        if depth[j, k] == 0:
            pass
        else:
            point_tmp = [0] * 3
            point_tmp[0] = (k - 320) / 619.444214 * depth[j, k] * 0.001
            point_tmp[1] = (j - 240) / 619.444336 * depth[j, k] * 0.001
            point_tmp[2] = depth[j, k] * 0.001
            # point_mask[3] = object_index
            # 在rpn阶段不需要cls的标签
            # 将像素坐标转换道 相机坐标系下 单位是m
            points.append(point_tmp)

fig = plt.figure()
ax = Axes3D(fig)
ax.set_title("instances_pcl")
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
print(np.array(points).shape)
ax.scatter3D(np.array(points)[:, 0], np.array(points)[:, 1], np.array(points)[:, 2], c="r", marker='.')

plt.show()



