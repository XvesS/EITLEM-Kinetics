from sklearn.metrics import mean_squared_error, r2_score
from tensorboard.backend.event_processing import event_accumulator
from collections import defaultdict
from scipy import stats
import numpy as np
import torch
import json
import os
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from matplotlib import rc
# 获取tensorboard数据函数
# def getTensorBoardData(path):
#     file_list = os.listdir(path)
#     data = defaultdict(dict)
#     for file_name in file_list:
#         if 'train' not in file_name and 'dev' not in file_name and 'test' not in file_name:
#             continue
#         info = file_name.split('_')
#         ea = event_accumulator.EventAccumulator(os.path.join(path, file_name))
#         ea.Reload()
#         data[info[1]][info[-1]] = [ i.value for i in ea.scalars.Items(info[-1])]
#     return data
    
def metric(pre, target, log10=True):
    if not log10:
        testY = np.log10(np.power(2, target))
        testPredict = np.log10(np.power(2, pre))
    else:
        testY = np.array(target)
        testPredict = np.array(pre)
    MAE = np.abs(testY - testPredict).sum() / len(pre)  # mean absolute error.
    rmse = np.sqrt(mean_squared_error(testY, testPredict))
    r2 = r2_score(testY, testPredict)
    correlation, p_value = stats.pearsonr(testY, testPredict)
    return {'MAE':MAE, 'rmse':rmse, 'r2':r2, 'r':correlation, 'p_value':p_value, 'samples':len(testY)}

def getPairInfo(low, high, index, pair, which=0):
    def getRangeIndex(low, high, index):
        rangIndex = []
        for k, v in index.items():
            if v >= low and v < high:
                rangIndex.append(k)
        return rangIndex
    pairIndex = getRangeIndex(low, high, index)
    pairInfo = []
    for item in pair:
        if item[which] in pairIndex:
            pairInfo.append(item)
    return pairInfo


def drawScatter(Real, Pre, Result, position, xlabel, ylabel, save_path):
    plt.figure(figsize=(5,5))
    rc('font',**{'family':'Times New Roman'})
    plt.rcParams['pdf.fonttype'] = 42

    plt.tick_params(direction='in')
    plt.tick_params(which='major',length=1.5)
    plt.tick_params(which='major',width=0.4)
    vstack = np.vstack([Real,Pre])
    experimental_predicted = gaussian_kde(vstack)(vstack)
    ax = plt.scatter(x = Real, y = Pre, c=experimental_predicted, s=3, cmap='rainbow')
    cbar = plt.colorbar(ax)
    cbar.ax.tick_params(labelsize=11)
    cbar.set_label('Density', size=12)

    plt.text(position[0][0], position[0][1], '$R^{2}$ = %.3f' % Result['r2'], fontweight ="normal", fontsize=11)
    plt.text(position[1][0], position[1][1], 'P.C.C = %.3f' % Result['r'], fontweight ="normal", fontsize=11)
    plt.text(position[2][0], position[2][1], 'RMSE = %.3f' % Result['rmse'], fontweight ="normal", fontsize=11)
    plt.text(position[3][0], position[3][1], f'N = {len(Real)}', fontweight ="normal", fontsize=11)

    plt.xlabel(xlabel, fontdict={'weight': 'bold', 'fontname': 'Times New Roman', 'size': 12})
    plt.ylabel(ylabel,fontdict={'weight': 'bold', 'fontname': 'Times New Roman', 'size': 12})

    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)

    # plt.ylim(-4)
    # plt.xlim(-7)

    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)

    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.show()