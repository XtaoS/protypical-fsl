import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
import numpy as np
import os
import math
import argparse
import scipy as sp
import scipy.stats
import pickle
import random
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn import metrics
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import time
import utils
import models
from thop import profile
import spectral

# np.random.seed(1337)

parser = argparse.ArgumentParser(description="Few Shot Visual Recognition")
parser.add_argument("-f","--feature_dim",type = int, default = 160)
parser.add_argument("-c","--src_input_dim",type = int, default = 128)
parser.add_argument("-d","--tar_input_dim",type = int, default = 103) # PaviaU=103；salinas=204
parser.add_argument("-n","--n_dim",type = int, default = 100)
parser.add_argument("-w","--class_num",type = int, default = 9)
parser.add_argument("-s","--shot_num_per_class",type = int, default = 1)
parser.add_argument("-b","--query_num_per_class",type = int, default = 19)
parser.add_argument("-e","--episode",type = int, default= 10000)
parser.add_argument("-t","--test_episode", type = int, default = 600)
parser.add_argument("-l","--learning_rate", type = float, default = 0.001)
parser.add_argument("-g","--gpu",type=int, default=0)
parser.add_argument("-u","--hidden_unit",type=int,default=10)
# target
parser.add_argument("-m","--test_class_num",type=int, default=9)
parser.add_argument("-z","--test_lsample_num_per_class",type=int,default=9, help='5 4 3 2 1')

args = parser.parse_args(args=[])

# Hyper Parameters
FEATURE_DIM = args.feature_dim
SRC_INPUT_DIMENSION = args.src_input_dim
TAR_INPUT_DIMENSION = args.tar_input_dim
N_DIMENSION = args.n_dim
CLASS_NUM = args.class_num
SHOT_NUM_PER_CLASS = args.shot_num_per_class
QUERY_NUM_PER_CLASS = args.query_num_per_class
EPISODE = args.episode
TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu
HIDDEN_UNIT = args.hidden_unit

# Hyper Parameters in target domain data set
TEST_CLASS_NUM = args.test_class_num  # the number of class
TEST_LSAMPLE_NUM_PER_CLASS = args.test_lsample_num_per_class  # the number of labeled samples per class 5 4 3 2 1


# utils.same_seeds(1)
def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('classificationMap'):
        os.makedirs('classificationMap')


_init_()

# load source domain data set
with open(os.path.join('datasets', 'Chikusei_imdb_128.pickle'), 'rb') as handle:
    source_imdb = pickle.load(handle)
print(source_imdb.keys())
print(source_imdb['Labels'])

# process source domain data set
data_train = source_imdb['data']  # (77592, 9, 9, 128)
labels_train = source_imdb['Labels']  # 77592
print(data_train.shape)
print(labels_train.shape)
keys_all_train = sorted(list(set(labels_train)))  # class [0,...,18]
print(keys_all_train)  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
label_encoder_train = {}
for i in range(len(keys_all_train)):
    label_encoder_train[keys_all_train[i]] = i
print(label_encoder_train)
# {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18}
train_set = {}
for class_, path in zip(labels_train, data_train):
    if label_encoder_train[class_] not in train_set:
        train_set[label_encoder_train[class_]] = []
    train_set[label_encoder_train[class_]].append(path)
print(train_set.keys())
# data = train_set
# del train_set
# del keys_all_train
# del label_encoder_train
# #
# print("Num classes for source domain datasets: " + str(len(data)))
# print(data.keys())
# data = utils.sanity_check(data) # 200 labels samples per class 为每个源域类取其前200个样本
# print("Num classes of the number of class larger than 200: " + str(len(data)))


# for class_ in data:
#     for i in range(len(data[class_])):
#         image_transpose = np.transpose(data[class_][i], (2, 0, 1))  # （9,9,128）-> (128,9,9)
#         data[class_][i] = image_transpose
#
# # source few-shot classification data
# metatrain_data = data
# print(len(metatrain_data.keys()), metatrain_data.keys())
# del data
#
# source domain adaptation data
print(source_imdb['data'].shape)  # (77592, 9, 9, 128)
source_imdb['data'] = source_imdb['data'].transpose((1, 2, 3, 0))  # (9, 9, 128, 77592)
print(source_imdb['data'].shape)  # (9, 9, 128, 77592)
print(source_imdb['Labels'])
source_dataset = utils.matcifar(source_imdb, train=True, d=3, medicinal=0)
source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=128, shuffle=True, num_workers=0)
del source_dataset, source_imdb
#
## target domain data set
# load target domain data set
test_data = 'datasets/paviaU/paviaU.mat'
test_label = 'datasets/paviaU/paviaU_gt.mat'

Data_Band_Scaler, GroundTruth = utils.load_data(test_data, test_label)


# get train_loader and test_loader
def get_train_test_loader(Data_Band_Scaler, GroundTruth, class_num, shot_num_per_class):
    print(Data_Band_Scaler.shape)  # (610, 340, 103)
    [nRow, nColumn, nBand] = Data_Band_Scaler.shape

    '''label start'''
    num_class = int(np.max(GroundTruth))
    data_band_scaler = utils.flip(Data_Band_Scaler)
    groundtruth = utils.flip(GroundTruth)
    print(data_band_scaler.shape)  # 1830, 1020, 103
    print(groundtruth.shape)  # 1830, 1020
    del Data_Band_Scaler
    del GroundTruth

    HalfWidth = 4
    G = groundtruth[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth]
    data = data_band_scaler[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth, :]
    print(G.shape)  # 618, 348
    print(data.shape)  # 618, 348, 103
    [Row, Column] = np.nonzero(G)  # 返回G中非零元素的位置
    print(Row)
    print(Column)
    del data_band_scaler
    del groundtruth

    nSample = np.size(Row)
    print('number of sample', nSample)  # pavia中共422776个样本

    # Sampling samples
    train = {}
    test = {}
    da_train = {}  # Data Augmentation
    m = int(np.max(G))  # 9
    nlabeled = TEST_LSAMPLE_NUM_PER_CLASS  # 5
    print('labeled number per class:', nlabeled)
    print((200 - nlabeled) / nlabeled + 1)
    print(math.ceil((200 - nlabeled) / nlabeled) + 1)  # 向上取整

    for i in range(m):
        indices = [j for j, x in enumerate(Row.ravel().tolist()) if
                   G[Row[j], Column[j]] == i + 1]  # ravel 将多维数组转化为一维数组 tolist将数组转化为列表
        # print(indices)
        np.random.shuffle(indices)  # 打乱顺序
        nb_val = shot_num_per_class  # 5
        train[i] = indices[:nb_val]  # 取每个类的前5个作为目标域样本
        da_train[i] = []
        for j in range(math.ceil((200 - nlabeled) / nlabeled) + 1):
            da_train[i] += indices[:nb_val]  # 取9个类，
        test[i] = indices[nb_val:]
    print(len(da_train))
    print(len(test))
    train_indices = []
    test_indices = []
    da_train_indices = []
    for i in range(m):
        train_indices += train[i]
        test_indices += test[i]
        da_train_indices += da_train[i]
    np.random.shuffle(test_indices)

    print('the number of train_indices:', len(train_indices))  # 45
    print('the number of test_indices:', len(test_indices))  # 42731
    print('the number of train_indices after data argumentation:', len(da_train_indices))  # 1800
    print('labeled sample indices:', train_indices)

    nTrain = len(train_indices)  # 45
    nTest = len(test_indices)  # 42731
    da_nTrain = len(da_train_indices)  # 1800

    imdb = {}
    imdb['data'] = np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, nTrain + nTest],
                            dtype=np.float32)  # (9,9,103,42776)
    imdb['Labels'] = np.zeros([nTrain + nTest], dtype=np.int64)  # 42776
    imdb['set'] = np.zeros([nTrain + nTest], dtype=np.int64)  # 42776

    RandPerm = train_indices + test_indices

    RandPerm = np.array(RandPerm)

    for iSample in range(nTrain + nTest):
        imdb['data'][:, :, :, iSample] = data[
                                         Row[RandPerm[iSample]] - HalfWidth:  Row[RandPerm[iSample]] + HalfWidth + 1,
                                         Column[RandPerm[iSample]] - HalfWidth: Column[
                                                                                    RandPerm[iSample]] + HalfWidth + 1,
                                         :]
        imdb['Labels'][iSample] = G[Row[RandPerm[iSample]], Column[RandPerm[iSample]]].astype(np.int64)

    imdb['Labels'] = imdb['Labels'] - 1  # 1-16 0-15
    imdb['set'] = np.hstack((np.ones([nTrain]), 3 * np.ones([nTest]))).astype(np.int64)
    print('Data is OK.')

    train_dataset = utils.matcifar(imdb, train=True, d=3, medicinal=0)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=class_num * shot_num_per_class, shuffle=False,
                                               num_workers=0)
    del train_dataset

    test_dataset = utils.matcifar(imdb, train=False, d=3, medicinal=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=0)
    del test_dataset
    del imdb

    # Data Augmentation for target domain for training
    imdb_da_train = {}
    imdb_da_train['data'] = np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, da_nTrain],
                                     dtype=np.float32)  # (9,9,100,n)
    imdb_da_train['Labels'] = np.zeros([da_nTrain], dtype=np.int64)
    imdb_da_train['set'] = np.zeros([da_nTrain], dtype=np.int64)

    da_RandPerm = np.array(da_train_indices)
    for iSample in range(da_nTrain):  # radiation_noise，flip_augmentation
        imdb_da_train['data'][:, :, :, iSample] = utils.radiation_noise(
            data[Row[da_RandPerm[iSample]] - HalfWidth:  Row[da_RandPerm[iSample]] + HalfWidth + 1,
            Column[da_RandPerm[iSample]] - HalfWidth: Column[da_RandPerm[iSample]] + HalfWidth + 1, :])
        imdb_da_train['Labels'][iSample] = G[Row[da_RandPerm[iSample]], Column[da_RandPerm[iSample]]].astype(np.int64)

    imdb_da_train['Labels'] = imdb_da_train['Labels'] - 1  # 1-16 0-15
    imdb_da_train['set'] = np.ones([da_nTrain]).astype(np.int64)
    print('ok')

    return train_loader, test_loader, imdb_da_train, G, RandPerm, Row, Column, nTrain


#          45，103，9，9   42731，103，9，9  9，9，103，1800   618*318  42776  42776  42776 R,C代表像素的位置 45
#
def get_target_dataset(Data_Band_Scaler, GroundTruth, class_num, shot_num_per_class):
    train_loader, test_loader, imdb_da_train, G, RandPerm, Row, Column, nTrain = get_train_test_loader(
        Data_Band_Scaler=Data_Band_Scaler, GroundTruth=GroundTruth, \
        class_num=class_num, shot_num_per_class=shot_num_per_class)  # 9 classes and 5 labeled samples per class
    train_datas, train_labels = train_loader.__iter__().next()
    print('train labels:', train_labels)
    print('size of train datas:', train_datas.shape)  # size of train datas: torch.Size([45, 103, 9, 9])

    print(imdb_da_train.keys())
    print(imdb_da_train['data'].shape)  # (9, 9, 103, 1800)
    print(imdb_da_train['Labels'])
    del Data_Band_Scaler, GroundTruth

    # target data with data augmentation
    target_da_datas = np.transpose(imdb_da_train['data'], (3, 2, 0, 1))  # (9,9,103, 1800)->(1800, 103, 9, 9)
    print(target_da_datas.shape)
    target_da_labels = imdb_da_train['Labels']  # (1800,)
    print('target data augmentation label:', target_da_labels)

    # metatrain data for few-shot classification
    target_da_train_set = {}
    for class_, path in zip(target_da_labels, target_da_datas):
        if class_ not in target_da_train_set:
            target_da_train_set[class_] = []
        target_da_train_set[class_].append(path)
    target_da_metatrain_data = target_da_train_set
    print(target_da_metatrain_data.keys())

    # target domain : batch samples for domian adaptation
    print(imdb_da_train['data'].shape)  # (9, 9, 100, 225)
    print(imdb_da_train['Labels'])
    target_dataset = utils.matcifar(imdb_da_train, train=True, d=3, medicinal=0)
    target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=128, shuffle=True, num_workers=0)
    del target_dataset

    return train_loader, test_loader, target_da_metatrain_data, target_loader, G, RandPerm, Row, Column, nTrain


class Spectral_attention(nn.Module):
    #  batchsize 16 25 200
    def __init__(self, in_features, hidden_features, out_features):
        super(Spectral_attention, self).__init__()
        self.conv1 = nn.Conv2d(in_features, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)
        # self.AvgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.MaxPool = nn.AdaptiveMaxPool2d((1, 1))
        self.SharedMLP = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, out_features),
            nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()  # ！

    def forward(self, X):
        # y1 = self.AvgPool(X)
        a1 = self.conv1(X)
        a2 = a1.view(a1.size(0), 1, -1)
        a3 = self.softmax(a2)
        a5 = torch.reshape(a3, (a3.shape[0], a3.shape[2], 1))
        a4 = X.view(X.size(0), X.size(1), -1)
        y1 = torch.matmul(a4, a5)
        y2 = self.MaxPool(X)
        y1 = y1.view(y1.size(0), -1)
        y2 = y2.view(y2.size(0), -1)
        # print(y1.shape, y2.shape)
        y1 = self.SharedMLP(y1)
        y2 = self.SharedMLP(y2)
        y = y1 + y2
        y = torch.reshape(y, (y.shape[0], y.shape[1], 1, 1))
        return self.sigmoid(y)  #


# class Spatial_attention(nn.Module):
#     # 2, 1, 3, 1, 1
#     def __init__(self, in_chanels, kernel_size, out_chanel, stride, padding):
#         super(Spatial_attention, self).__init__()
#         # self.AvgPool = nn.AdaptiveAvgPool2d((17, 17))  # (N, 200, 17, 17)
#         # self.MaxPool = nn.AdaptiveMaxPool2d((17, 17))
#         self.conv1 = nn.Conv2d(in_chanels, out_chanel, kernel_size=kernel_size, stride=stride, padding=padding)
#         self.act = nn.Sigmoid()
#
#     def forward(self, X):
#         # y1 = self.AvgPool(X)
#         # y2 = self.MaxPool(X)
#         avg_out = torch.mean(X, dim=1, keepdim=True)
#         max_out, _ = torch.max(X, dim=1, keepdim=True)
#         y = torch.cat((avg_out, max_out), 1)
#         y = self.conv1(y)
#         return self.act(y)

class RSSAN(nn.Module):
    def __init__(self, CLASS_NUM, in_chanels, kernel_size, out_chanel, stride, padding, windows, out_chanel1,
                 out_chanel2):
        # 16, 200, 3, 32, 1, 1
        super(RSSAN, self).__init__()
        self.attention1 = Spectral_attention(in_chanels, int(in_chanels // 8), in_chanels)
        # self.attention2 = Spatial_attention(2, 3, 1, 1, 1)
        self.conv1 = nn.Conv2d(in_chanels, out_chanel1, kernel_size=kernel_size, stride=stride, padding=padding)

        self.bn1 = nn.Sequential(
            nn.BatchNorm2d(out_chanel1, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Conv2d(out_chanel1, out_channels=out_chanel1, kernel_size=kernel_size,
                               stride=stride, padding=padding)

        self.bn2 = nn.Sequential(
            nn.BatchNorm2d(out_chanel1, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Conv2d(out_chanel1, out_chanel1, kernel_size=kernel_size,
                               stride=stride, padding=padding)
        self.bn3 = nn.BatchNorm2d(out_chanel1, eps=0.001, momentum=0.1, affine=True)
        self.attention3 = Spectral_attention(out_chanel1, out_chanel1 // 8, out_chanel1)
        # self.attention4 = Spatial_attention(2, 3, 1, 1, 1)
        self.relu1 = nn.ReLU()
        self.conv7 = nn.Conv2d(out_chanel1, out_chanel2, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn7 = nn.Sequential(
            nn.BatchNorm2d(out_chanel2, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Conv2d(out_chanel2, out_chanel2, kernel_size=kernel_size,
                               stride=stride, padding=padding)
        self.bn4 = nn.Sequential(
            nn.BatchNorm2d(out_chanel2, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )

        self.conv5 = nn.Conv2d(out_chanel2, out_chanel2, kernel_size=kernel_size,
                               stride=stride, padding=padding)
        self.bn5 = nn.BatchNorm2d(out_chanel2, eps=0.001, momentum=0.1, affine=True)
        self.attention5 = Spectral_attention(out_chanel2, out_chanel2 // 8, out_chanel2)
        # self.attention6 = Spatial_attention(2, 3, 1, 1, 1)
        self.relu2 = nn.ReLU()

        # self.conv8 = nn.Conv2d(out_chanel2, out_chanel3, kernel_size=kernel_size, stride=stride, padding=padding)
        # self.bn8 = nn.Sequential(
        #     nn.BatchNorm2d(out_chanel3, eps=0.001, momentum=0.1, affine=True),
        #     nn.ReLU(inplace=True),
        # )
        # self.conv9 = nn.Conv2d(out_chanel3, out_chanel3, kernel_size=kernel_size,
        #                        stride=stride, padding=padding)
        # self.bn9 = nn.Sequential(
        #     nn.BatchNorm2d(out_chanel3, eps=0.001, momentum=0.1, affine=True),
        #     nn.ReLU(inplace=True)
        # )
        #
        # self.conv10 = nn.Conv2d(out_chanel3, out_chanel3, kernel_size=kernel_size,
        #                         stride=stride, padding=padding)
        # self.bn10 = nn.BatchNorm2d(out_chanel3, eps=0.001, momentum=0.1, affine=True)
        # self.attention7 = Spectral_attention(out_chanel3, out_chanel3 // 8, out_chanel3)
        # # self.attention6 = Spatial_attention(2, 3, 1, 1, 1)
        # self.relu3 = nn.ReLU()

        self.avgpool = nn.AvgPool2d(kernel_size=(2, 2), padding=(1, 1), stride=(2, 2))
        # 1*1
        self.conv6 = nn.Conv2d(out_chanel2, out_chanel, kernel_size=(1, 1),
                               stride=stride, padding=0)

        # self.conv6 = nn.Conv2d(out_chanel3, out_chanel, kernel_size=(1, 1),       #单注意力
        #                        stride=stride, padding=0)

        self.full_connection = nn.Sequential(
            nn.Linear(out_chanel * windows * windows, CLASS_NUM),
            # nn.Sigmoid()
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)

    def forward(self, X):
        x1 = self.attention1(X)
        x3 = x1 * X
        # print(x3.shape)
        # x4 = self.attention2(x3) * x3

        x5 = self.conv1(x3)
        x6 = self.bn1(x5)  # #
        x7 = self.conv2(x6)
        x8 = self.bn2(x7)
        x9 = self.bn3(self.conv3(x8))  # #
        se = self.attention3(x9) * x9
        # sa = self.attention4(se) * se
        x10 = self.relu1(se + x6)  # #


        x16 = self.conv7(x10)
        x17 = self.bn7(x16)
        x11 = self.conv4(x17)
        x12 = self.bn4(x11)
        x13 = self.bn5(self.conv5(x12))  # #
        se1 = self.attention5(x13) * x13
        # sa1 = self.attention6(se1) * se1
        x14 = self.relu2(se1 + x17)

        # x18 = self.conv8(x14)
        # x19 = self.bn8(x18)
        # x20 = self.conv9(x19)
        # x21 = self.bn9(x20)
        # x22 = self.bn10(self.conv10(x21))  # #
        # se2 = self.attention7(x22) * x22
        # # sa1 = self.attention6(se1) * se1
        # x23 = self.relu2(se2 + x19)


        # print(x14.size())
        # x15 = self.conv6(self.avgpool(x23))
        x15 = self.conv6(self.avgpool(x14))
        y = x15.view(x15.size(0), -1)
        # print(x16.size())
        return y


class Mapping(nn.Module):
    def __init__(self, in_dimension, out_dimension):
        super(Mapping, self).__init__()
        self.preconv = nn.Conv2d(in_dimension, out_dimension, 1, 1, bias=False)
        self.preconv_bn = nn.BatchNorm2d(out_dimension)

    def forward(self, x):
        x = self.preconv(x)
        x = self.preconv_bn(x)
        return x


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.feature_model = RSSAN(CLASS_NUM, in_chanels=100, kernel_size=3, out_chanel=8, stride=1, padding=1,
                                   windows=5, out_chanel1=64, out_chanel2=32)
        self.target_mapping = Mapping(TAR_INPUT_DIMENSION, N_DIMENSION)  # 将Pavia从103维降为100维
        self.source_mapping = Mapping(SRC_INPUT_DIMENSION, N_DIMENSION)  # 将源数据集从128维降至100维 使用1*1的卷积核
        self.classifier = nn.Linear(in_features=8 * 5 * 5, out_features=CLASS_NUM)

    def forward(self, x, domain='source'):  # x
        if domain == 'target':
            x = self.target_mapping(x)  # (45, 100,9,9)
        elif domain == 'source':
            x = self.source_mapping(x)
        feature = self.feature_model(x)
        output = self.classifier(feature)
        return feature, output


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data = torch.ones(m.bias.data.size())


crossEntropy = nn.CrossEntropyLoss().cuda()
domain_criterion = nn.BCEWithLogitsLoss().cuda()


def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b) ** 2).sum(dim=2)
    return logits


# run 10 times
nDataSet = 1
acc = np.zeros([nDataSet, 1])
A = np.zeros([nDataSet, CLASS_NUM])
k = np.zeros([nDataSet, 1])
best_predict_all = []
best_acc_all = 0.0
best_G, best_RandPerm, best_Row, best_Column, best_nTrain = None, None, None, None, None

seeds = [1330]
for iDataSet in range(nDataSet):
    # load target domain data for training and testing
    # np.random.seed(seeds[iDataSet])
    train_loader, test_loader, target_da_metatrain_data, target_loader, G, RandPerm, Row, Column, nTrain = get_target_dataset(
        Data_Band_Scaler=Data_Band_Scaler, GroundTruth=GroundTruth, class_num=TEST_CLASS_NUM,
        shot_num_per_class=TEST_LSAMPLE_NUM_PER_CLASS)
    # model
    feature_encoder = Network()
    domain_classifier = models.DomainClassifier()
    random_layer = models.RandomLayer([200, args.class_num], 1024)
    #                                         160                9
    feature_encoder.apply(weights_init)
    domain_classifier.apply(weights_init)

    feature_encoder.cuda()
    domain_classifier.cuda()
    random_layer.cuda()  # Random layer

    feature_encoder.train()
    domain_classifier.train()
    # optimizer
    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(), lr=args.learning_rate)
    domain_classifier_optim = torch.optim.Adam(domain_classifier.parameters(), lr=args.learning_rate)

    print("Training...")

    last_accuracy = 0.0
    best_episdoe = 0
    train_loss = []
    test_acc = []
    running_D_loss, running_F_loss = 0.0, 0.0
    running_label_loss = 0
    # running_domain_loss = 0
    total_hit, total_num = 0.0, 0.0
    test_acc_list = []

    source_iter = iter(source_loader)
    target_iter = iter(target_loader)
    len_dataloader = min(len(source_loader), len(target_loader))
    train_start = time.time()
    for episode in range(20000):  # EPISODE = 90000
        # get domain adaptation data from  source domain and target domain
        try:
            source_data, source_label = source_iter.next()
        except Exception as err:
            source_iter = iter(source_loader)
            source_data, source_label = source_iter.next()

        try:
            target_data, target_label = target_iter.next()
        except Exception as err:
            target_iter = iter(target_loader)
            target_data, target_label = target_iter.next()

        # source domain few-shot + domain adaptation
        if episode % 2 == 0:
            '''Few-shot claification for source domain data set'''
            # get few-shot classification samples
            data = train_set
            data = utils.sanity_check(data)  # 200 labels samples per class
            for class_ in data:
                for i in range(len(data[class_])):
                    image_transpose = np.transpose(data[class_][i], (2, 0, 1))  # （9,9,128）-> (128,9,9)
                    data[class_][i] = image_transpose

            # source few-shot classification data
            metatrain_data = data
            del data
            # # print(len(metatrain_data.keys()), metatrain_data.keys())
            # del data
            task = utils.Task(metatrain_data, CLASS_NUM, SHOT_NUM_PER_CLASS, QUERY_NUM_PER_CLASS)  # 5， 1，15
            del metatrain_data
            support_dataloader = utils.get_HBKC_data_loader(task, num_per_class=SHOT_NUM_PER_CLASS, split="train",
                                                            shuffle=False)
            query_dataloader = utils.get_HBKC_data_loader(task, num_per_class=QUERY_NUM_PER_CLASS, split="test",
                                                          shuffle=True)

            # sample datas
            supports, support_labels = support_dataloader.__iter__().next()  # (5, 100, 9, 9)
            querys, query_labels = query_dataloader.__iter__().next()  # (75,100,9,9)

            # calculate features
            support_features, support_outputs = feature_encoder(supports.cuda())  # torch.Size([409, 32, 7, 3, 3])
            query_features, query_outputs = feature_encoder(querys.cuda())  # torch.Size([409, 32, 7, 3, 3])
            target_features, target_outputs = feature_encoder(target_data.cuda(),
                                                              domain='target')  # torch.Size([409, 32, 7, 3, 3])

            # Prototype network
            if SHOT_NUM_PER_CLASS > 1:
                support_proto = support_features.reshape(CLASS_NUM, SHOT_NUM_PER_CLASS, -1).mean(dim=1)  # (9, 160)
            else:
                support_proto = support_features

            # fsl_loss
            logits = euclidean_metric(query_features, support_proto)
            f_loss = crossEntropy(logits, (query_labels.cuda()).long())
            output_query_logits = logits
            '''domain adaptation'''
            # calculate domain adaptation loss
            features = torch.cat([query_features, target_features], dim=0)
            output_target_logits = euclidean_metric(target_features, support_proto)
            outputs = torch.cat((output_query_logits, output_target_logits), dim=0)
            softmax_output = nn.Softmax(dim=1)(outputs)

            # set label: source 1; target 0
            domain_label = torch.zeros([querys.shape[0] + target_data.shape[0], 1]).cuda()
            domain_label[:querys.shape[0]] = 1  # torch.Size([225=9*20+9*4, 100, 9, 9])

            randomlayer_out = random_layer.forward([features, softmax_output])  # torch.Size([225, 1024=32*7*3*3])

            domain_logits = domain_classifier(randomlayer_out, episode)
            domain_loss = domain_criterion(domain_logits, domain_label)

            # total_loss = fsl_loss + domain_loss
            loss = f_loss + domain_loss  # 0.01
            # loss = f_loss
            # Update parameters
            feature_encoder.zero_grad()
            domain_classifier.zero_grad()
            loss.backward()
            feature_encoder_optim.step()
            domain_classifier_optim.step()

            total_hit += torch.sum(torch.argmax(logits, dim=1).cpu() == query_labels).item()
            total_num += querys.shape[0]
        # target domain few-shot + domain adaptation
        else:
            '''Few-shot classification for target domain data set'''
            # get few-shot classification samples
            task = utils.Task(target_da_metatrain_data, TEST_CLASS_NUM, SHOT_NUM_PER_CLASS,
                              QUERY_NUM_PER_CLASS)  # 5， 1，15
            support_dataloader = utils.get_HBKC_data_loader(task, num_per_class=SHOT_NUM_PER_CLASS, split="train",
                                                            shuffle=False)
            query_dataloader = utils.get_HBKC_data_loader(task, num_per_class=QUERY_NUM_PER_CLASS, split="test",
                                                          shuffle=True)

            # sample datas
            supports, support_labels = support_dataloader.__iter__().next()  # (5, 100, 9, 9)
            querys, query_labels = query_dataloader.__iter__().next()  # (75,100,9,9)

            # calculate features
            support_features, support_outputs = feature_encoder(supports.cuda(),
                                                                domain='target')  # torch.Size([409, 32, 7, 3, 3])
            query_features, query_outputs = feature_encoder(querys.cuda(),
                                                            domain='target')  # torch.Size([409, 32, 7, 3, 3])
            source_features, source_outputs = feature_encoder(source_data.cuda())  # torch.Size([409, 32, 7, 3, 3])

            # Prototype network
            if SHOT_NUM_PER_CLASS > 1:
                support_proto = support_features.reshape(CLASS_NUM, SHOT_NUM_PER_CLASS, -1).mean(dim=1)  # (9, 160)
            else:
                support_proto = support_features

            # fsl_loss
            logits = euclidean_metric(query_features, support_proto)
            f_loss = crossEntropy(logits, (query_labels.cuda()).long())
            output_query_logits = logits
            source_outputs = euclidean_metric(source_features, support_proto)
            '''domain adaptation'''
            features = torch.cat([query_features, source_features], dim=0)
            outputs = torch.cat((output_query_logits, source_outputs), dim=0)
            softmax_output = nn.Softmax(dim=1)(outputs)

            #
            domain_label = torch.zeros([querys.shape[0] + source_features.shape[0], 1]).cuda()
            domain_label[querys.shape[0]:] = 1

            randomlayer_out = random_layer.forward([features, softmax_output])  # torch.Size([225, 1024=32*7*3*3])

            domain_logits = domain_classifier(randomlayer_out, episode)  # , label_logits
            domain_loss = domain_criterion(domain_logits, domain_label)

            # total_loss = fsl_loss + domain_loss
            loss = f_loss + domain_loss  # 0.01 0.5=78;0.25=80;0.01=80

            # Update parameters
            feature_encoder.zero_grad()
            domain_classifier.zero_grad()
            loss.backward()
            feature_encoder_optim.step()
            domain_classifier_optim.step()

            total_hit += torch.sum(torch.argmax(logits, dim=1).cpu() == query_labels).item()
            total_num += querys.shape[0]

        if (episode + 1) % 100 == 0:  # display
            train_loss.append(loss.item())
            print('episode {:>3d}, fsl loss: {:6.4f}, acc {:6.4f}, loss: {:6.4f}'.format(episode + 1, \
                                                                                         f_loss.item(),
                                                                                         total_hit / total_num,
                                                                                         loss.item()))

        if (episode + 1) % 1000 == 0 or episode == 0:
            # test
            print("Testing ...")
            train_end = time.time()

            feature_encoder.eval()
            total_rewards = 0
            counter = 0
            accuracies = []
            predict = np.array([], dtype=np.int64)
            labels = np.array([], dtype=np.int64)

            train_datas, train_labels = train_loader.__iter__().next()
            train_features, _ = feature_encoder(Variable(train_datas).cuda(), domain='target')  # (45, 160)

            max_value = train_features.max()  # 89.67885
            min_value = train_features.min()  # -57.92479
            print(max_value.item())
            print(min_value.item())
            train_features = (train_features - min_value) * 1.0 / (max_value - min_value)

            KNN_classifier = KNeighborsClassifier(n_neighbors=1)
            KNN_classifier.fit(train_features.cpu().detach().numpy(), train_labels)  # .cpu().detach().numpy()
            for test_datas, test_labels in test_loader:
                batch_size = test_labels.shape[0]

                test_features, _ = feature_encoder(Variable(test_datas).cuda(), domain='target')  # (100, 160)
                test_features = (test_features - min_value) * 1.0 / (max_value - min_value)
                predict_labels = KNN_classifier.predict(test_features.cpu().detach().numpy())
                test_labels = test_labels.numpy()
                rewards = [1 if predict_labels[j] == test_labels[j] else 0 for j in range(batch_size)]

                total_rewards += np.sum(rewards)
                counter += batch_size

                predict = np.append(predict, predict_labels)
                labels = np.append(labels, test_labels)

                accuracy = total_rewards / 1.0 / counter  #
                accuracies.append(accuracy)

            test_accuracy = 100. * total_rewards / len(test_loader.dataset)

            print('\t\tAccuracy: {}/{} ({:.2f}%)\n'.format(total_rewards, len(test_loader.dataset),
                                                           100. * total_rewards / len(test_loader.dataset)))
            test_end = time.time()

            # Training mode
            feature_encoder.train()
            if test_accuracy > last_accuracy:
                # save networks
                torch.save(feature_encoder.state_dict(),
                           str("checkpoints/DFSL_feature_encoder_" + "UP_" + str(iDataSet) + "iter_" + str(
                               TEST_LSAMPLE_NUM_PER_CLASS) + "shot.pkl"))
                print("save networks for episode:", episode + 1)
                last_accuracy = test_accuracy
                best_episdoe = episode

                acc[iDataSet] = 100. * total_rewards / len(test_loader.dataset)
                OA = acc
                C = metrics.confusion_matrix(labels, predict)
                A[iDataSet, :] = np.diag(C) / np.sum(C, 1, dtype=np.float)
                print("**********************************************************************************************")
                print(A[iDataSet, :])
                k[iDataSet] = metrics.cohen_kappa_score(labels, predict)

            print('best episode:[{}], best accuracy={}'.format(best_episdoe + 1, last_accuracy))
            print("train time per DataSet(s): " + "{:.5f}".format(train_end - train_start))
    if test_accuracy > best_acc_all:
        best_predict_all = predict
        best_G, best_RandPerm, best_Row, best_Column, best_nTrain = G, RandPerm, Row, Column, nTrain
    print('iter:{} best episode:[{}], best accuracy={}'.format(iDataSet, best_episdoe + 1, last_accuracy))
    print('***********************************************************************************')
    print(OA)
    print(k)
    print(A)

AA = np.mean(A, 1)

AAMean = np.mean(AA, 0)
AAStd = np.std(AA)

AMean = np.mean(A, 0)
AStd = np.std(A, 0)

OAMean = np.mean(acc)
OAStd = np.std(acc)

kMean = np.mean(k)
kStd = np.std(k)
print("train time per DataSet(s): " + "{:.5f}".format(train_end - train_start))
print("test time per DataSet(s): " + "{:.5f}".format(test_end - train_end))
print("average OA: " + "{:.2f}".format(OAMean) + " +- " + "{:.2f}".format(OAStd))
print("average AA: " + "{:.2f}".format(100 * AAMean) + " +- " + "{:.2f}".format(100 * AAStd))
print("average kappa: " + "{:.4f}".format(100 * kMean) + " +- " + "{:.4f}".format(100 * kStd))
print("accuracy for each class: ")
for i in range(CLASS_NUM):
    print("Class " + str(i) + ": " + "{:.2f}".format(100 * AMean[i]) + " +- " + "{:.2f}".format(100 * AStd[i]))

# little = 0
# for i in range(10):
#     for j in range(acc.shape[0]):
#         if acc[j] == min(acc):
#             little = j
#     del acc[little]
#     del A[little, :]
#     del k[little]

# AA = np.mean(A, 1)
#
# AAMean = np.mean(AA,0)
# AAStd = np.std(AA)
#
# AMean = np.mean(A, 0)
# AStd = np.std(A, 0)
#
# OAMean = np.mean(acc)
# OAStd = np.std(acc)
#
# kMean = np.mean(k)
# kStd = np.std(k)
# print("train time per DataSet(s): " + "{:.5f}".format(train_end-train_start))
# print("test time per DataSet(s): " + "{:.5f}".format(test_end-train_end))
# print("average OA: " + "{:.2f}".format( OAMean) + " +- " + "{:.2f}".format(OAStd))
# print("average AA: " + "{:.2f}".format(100 * AAMean) + " +- " + "{:.2f}".format(100 * AAStd))
# print("average kappa: " + "{:.4f}".format(100 *kMean) + " +- " + "{:.4f}".format(100 *kStd))
# print("accuracy for each class: ")
# for i in range(CLASS_NUM):
#     print("Class " + str(i) + ": " + "{:.2f}".format(100 * AMean[i]) + " +- " + "{:.2f}".format(100 * AStd[i]))


best_iDataset = 0
for i in range(len(acc)):
    print('{}:{}'.format(i, acc[i]))
    if acc[i] > acc[best_iDataset]:
        best_iDataset = i
print('best acc all={}'.format(acc[best_iDataset]))
# print(A)
#################classification map################################

# print(acc)
# print(len(acc))
last_iDataset = 0
acc1 = np.zeros([10])
A1 = np.zeros([10, CLASS_NUM])
k1 = np.zeros([10])
for j in range(10):
    for i in range(len(acc)):
        if acc[i] >= acc[last_iDataset]:
            last_iDataset = i
    acc1[j] = acc[last_iDataset, :]
    A1[j, :] = A[last_iDataset, :]
    k1[j] = k[last_iDataset]
    acc[last_iDataset] = 0
    A[last_iDataset, :] = 0
    k[last_iDataset] = 0
# print(acc1)
# print(k1)


AA1 = np.mean(A1, 1)

AAMean1 = np.mean(AA1, 0)
AAStd1 = np.std(AA1)

AMean1 = np.mean(A1, 0)
AStd1 = np.std(A1, 0)

OAMean1 = np.mean(acc1)
OAStd1 = np.std(acc1)

kMean1 = np.mean(k1)
kStd1 = np.std(k1)
print("****************************10 times classification**********************************************")
print("average OA: " + "{:.2f}".format(OAMean1) + " +- " + "{:.2f}".format(OAStd1))
print("average AA: " + "{:.2f}".format(100 * AAMean1) + " +- " + "{:.2f}".format(100 * AAStd1))
print("average kappa: " + "{:.4f}".format(100 * kMean1) + " +- " + "{:.4f}".format(100 * kStd1))
print("accuracy for each class: ")
for i in range(CLASS_NUM):
    print("Class " + str(i) + ": " + "{:.2f}".format(100 * AMean1[i]) + " +- " + "{:.2f}".format(100 * AStd1[i]))
best_iDataset = 0
for i in range(len(acc1)):
    print('{}:{}'.format(i, acc1[i]))
    if acc1[i] > acc1[best_iDataset]:
        best_iDataset = i
print('best acc all={}'.format(acc1[best_iDataset]))
for i in range(len(acc1)):
    print('{}:{}'.format(i, k1[i]))
print(A1)

for i in range(len(best_predict_all)):  # predict ndarray <class 'tuple'>: (9729,)
    best_G[best_Row[best_RandPerm[best_nTrain + i]]][best_Column[best_RandPerm[best_nTrain + i]]] = best_predict_all[
                                                                                                        i] + 1

hsi_pic = np.zeros((best_G.shape[0], best_G.shape[1], 3))

for i in range(best_G.shape[0]):
    for j in range(best_G.shape[1]):
        if best_G[i][j] == 0:
            hsi_pic[i, j, :] = [0, 0, 0]
        if best_G[i][j] == 1:
            hsi_pic[i, j, :] = [0, 0, 1]
        if best_G[i][j] == 2:
            hsi_pic[i, j, :] = [0, 1, 0]
        if best_G[i][j] == 3:
            hsi_pic[i, j, :] = [0, 1, 1]
        if best_G[i][j] == 4:
            hsi_pic[i, j, :] = [1, 0, 0]
        if best_G[i][j] == 5:
            hsi_pic[i, j, :] = [1, 0, 1]
        if best_G[i][j] == 6:
            hsi_pic[i, j, :] = [1, 1, 0]
        if best_G[i][j] == 7:
            hsi_pic[i, j, :] = [0.5, 0.5, 1]
        if best_G[i][j] == 8:
            hsi_pic[i, j, :] = [0.65, 0.35, 1]
utils.classification_map(hsi_pic[4:-4, 4:-4, :], best_G[4:-4, 4:-4], 24,
                         "classificationMap/IP_{}shot.png".format(TEST_LSAMPLE_NUM_PER_CLASS))
