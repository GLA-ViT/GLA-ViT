import train
import argparse
import os

server = dict()
server['lab'] = {
    'default_img_train': r'',
    'default_img_test': r'',
    'default_text_path': r'',
    'log_dir': r''
}
default_path = server['lab']
parser = argparse.ArgumentParser(description="Training and Testing Script")
# 添加模式参数,train包含了test
parser.add_argument('--mode', action='store', dest='mode', required=True, type=str, choices=['train', 'test'],
                    help="Mode: train or test")
# 分类数、batch size等参数
parser.add_argument('--CLS', action='store', dest='CLS', required=True, type=int)  # 分类数
parser.add_argument('--BSZ', action='store', dest='BSZ', required=True, type=int)  # batch size
parser.add_argument('--TRAIN_IMG_DIR', action='store', dest='TRAIN_IMG_DIR', type=str,
                    help='Train image directory', default=default_path['default_img_train'])  # 训练图像目录
parser.add_argument('--TEST_IMG_DIR', action='store', dest='TEST_IMG_DIR', type=str,
                    help='TEST image directory', default=default_path['default_img_test'])  # 测试图像目录
default_text_dir = default_path['default_text_path']
parser.add_argument('--TRAIN_DIR', action='store', dest='TRAIN_DIR', type=str, help='Train data directory',
                    default=os.path.join(default_text_dir, 'train.csv'))  # 训练文本路径
parser.add_argument('--TEST_DIR', action='store', dest='TEST_DIR', type=str, help='Test data directory',
                    default=os.path.join(default_text_dir, 'test.csv'))  # 测试文本路径

args = parser.parse_args()
log_dir = default_path['log_dir']
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
num_epochs = 120
# 根据模式选择训练或测试
if args.mode == 'train':
    train(args)
