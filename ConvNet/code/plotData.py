from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

font = {'family': 'serif',
        'color': 'darkred',
        'weight': 'normal',
        'size': 16,
        }

# avgPoolAdam =pd.read_csv('../data/avgPool_adam.csv', header = None, quoting=3, sep=',').as_matrix()
# avgPoolMom =pd.read_csv('../data/avgPool_momentumSGD.csv', header = None, quoting=3, sep=',').as_matrix()
# avgPoolRMS =pd.read_csv('../data/avgPool_rmsProp.csv', header = None, quoting=3, sep=',').as_matrix()
#
# maxPoolAdam =pd.read_csv('../data/maxPool_adam.csv', header = None, quoting=3, sep=',').as_matrix()
# maxPoolMom =pd.read_csv('../data/baseline.csv', header = None, quoting=3, sep=',').as_matrix()
# maxPoolRMS =pd.read_csv('../data/maxPool_RMSProp.csv', header = None, quoting=3, sep=',').as_matrix()
#
# fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
#
# with plt.style.context('fivethirtyeight'):
#   axes[0].plot(avgPoolAdam[:,0], 100 - avgPoolAdam[:,1],  '--', color='blue', label = 'Adam')
#   axes[0].plot(avgPoolMom[:,0], 100 - avgPoolMom[:,1], '--', color = 'red', label = 'MomentumSGD')
#   axes[0].plot(avgPoolRMS[:, 0], 100 - avgPoolRMS[:, 1], '--', color = 'green', label = 'RMSProp')
#   axes[1].plot(maxPoolAdam[0:15,0], 100 - maxPoolAdam[0:15,1],  '--', color='blue', label = 'Adam')
#   axes[1].plot(maxPoolMom[0:15,0], 100 - maxPoolMom[0:15,1], '--', color = 'red', label = 'MomentumSGD')
#   axes[1].plot(maxPoolRMS[0:15, 0], 100 - maxPoolRMS[0:15, 1], '--', color = 'green', label = 'RMSProp')
# axes[0].legend(loc = 'best')
# axes[1].legend(loc = 'best')
# axes[0].set_title("Average Pooling")
# axes[1].set_title("Max Pooling")
# axes[0].set_xlabel('Number of Epochs')
# axes[0].set_ylabel('Train Error')
# axes[1].set_xlabel('Number of Epochs')
# axes[1].set_ylabel('Train Error')
# plt.setp(axes)
# plt.show()


# with plt.style.context('fivethirtyeight'):
#   axes[0].plot(avgPoolAdam[:,0], avgPoolAdam[:,2],  '--', color='blue', label = 'Adam')
#   axes[0].plot(avgPoolMom[:,0], avgPoolMom[:,2], '--', color = 'red', label = 'MomentumSGD')
#   axes[0].plot(avgPoolRMS[:, 0], avgPoolRMS[:, 2], '--', color = 'green', label = 'RMSProp')
#   axes[1].plot(maxPoolAdam[0:15,0], maxPoolAdam[0:15,2],  '--', color='blue', label = 'Adam')
#   axes[1].plot(maxPoolMom[0:15,0], maxPoolMom[0:15,2], '--', color = 'red', label = 'MomentumSGD')
#   axes[1].plot(maxPoolRMS[0:15, 0], maxPoolRMS[0:15, 2], '--', color = 'green', label = 'RMSProp')
# axes[0].legend(loc = 'best')
# axes[1].legend(loc = 'best')
# axes[0].set_title("Average Pooling")
# axes[1].set_title("Max Pooling")
# axes[0].set_xlabel('Number of Epochs')
# axes[0].set_ylabel('Test Accuracy')
# axes[1].set_xlabel('Number of Epochs')
# axes[1].set_ylabel('Test Accuracy')
# plt.setp(axes)
# plt.show()


# plt.legend(loc = 'best')
# plt.title("Average Pooling")
# plt.ylabel("Train Error")
# plt.xlabel("Number of epochs")
# plt.show()

#
# with plt.style.context('fivethirtyeight'):
#   plt.plot(maxPoolAdam[0:15,0], 100 - maxPoolAdam[0:15,1],  '--', color='blue', label = 'Adam')
#   plt.plot(maxPoolMom[0:15,0], 100 - maxPoolMom[0:15,1], '--', color = 'red', label = 'MomentumSGD')
#   plt.plot(maxPoolRMS[0:15, 0], 100 - maxPoolRMS[0:15, 1], '--', color = 'green', label = 'RMSProp')
# plt.legend(loc = 'center right')
# plt.title("Max Pooling")
# plt.ylabel("Train Error")
# plt.xlabel("Number of epochs")
# plt.show()

# noDropout =pd.read_csv('../data/noDropout.csv', header = None, quoting=3, sep=',').as_matrix()
# dropout_3 =pd.read_csv('../data/dropout_0.3.csv', header = None, quoting=3, sep=',').as_matrix()
# dropout_4 =pd.read_csv('../data/dropout_0.4.csv', header = None, quoting=3, sep=',').as_matrix()
# dropout_5 =pd.read_csv('../data/dropout_0.5.csv', header = None, quoting=3, sep=',').as_matrix()
# with plt.style.context('fivethirtyeight'):
#   plt.plot(noDropout[0:20,0], 100 - noDropout[0:20,1], '--', color='blue', label = 'no dropout')
#   plt.plot(dropout_3[0:20,0], 100 - dropout_3[0:20,1], '--', color='red', label = 'dropout =  0.3')
#   plt.plot(dropout_4[0:20, 0], 100 - dropout_4[0:20, 1], '--', color='green', label = 'dropout =  0.4')
#   plt.plot(dropout_5[0:20, 0], 100 - dropout_5[0;20, 1], '--', color='orange', label = 'dropout =  0.5')
# plt.legend(loc = 'upper right')
# plt.title("Max-pooling Dropout")
# plt.ylabel("Train Error")
# plt.xlabel("Number of epochs")
# plt.show()


# noDropout =pd.read_csv('../data/noDropout.csv', header = None, quoting=3, sep=',').as_matrix()
# dropout_25 =pd.read_csv('../data/dropout_0.25.csv', header = None, quoting=3, sep=',').as_matrix()
# dropout_3 =pd.read_csv('../data/dropout_0.3.csv', header = None, quoting=3, sep=',').as_matrix()
# dropout_4 =pd.read_csv('../data/dropout_0.4.csv', header = None, quoting=3, sep=',').as_matrix()
# dropout_5 =pd.read_csv('../data/dropout_0.5.csv', header = None, quoting=3, sep=',').as_matrix()
#
# with plt.style.context('fivethirtyeight'):
#    plt.plot(noDropout[0:20,0], 100 - noDropout[0:20,2], '--', color='blue', label = 'no dropout')
#    plt.plot(dropout_25[0:20,0], 100 - dropout_25[0:20,2], '--', color='yellow', label = 'dropout =  0.25')
#    plt.plot(dropout_3[0:20,0], 100 - dropout_3[0:20,2], '--', color='red', label = 'dropout =  0.3')
#    plt.plot(dropout_4[0:20, 0], 100 - dropout_4[0:20,2], '--', color='green', label = 'dropout =  0.4')
#    plt.plot(dropout_5[0:20, 0], 100 - dropout_5[0:20, 2], '--', color = 'orange', label = 'dropout =  0.5')
# plt.legend(loc = 'upper right')
# plt.title("Max-pooling Dropout")
# plt.ylabel("Test Error")
# plt.xlabel("Number of epochs")
# plt.show()

# baseline  = pd.read_csv('../data/baseline.csv', header = None, quoting=3, sep = ',').as_matrix()
# noBN = pd.read_csv('../data/noBatchNormalization.csv', header = None, quoting=3, sep = ',').as_matrix()
# convA  = pd.read_csv('../data/convA.csv', header = None, quoting=3, sep = ',').as_matrix()
# convE  = pd.read_csv('../data/convE.csv', header = None, quoting=3, sep = ',').as_matrix()
# #
# with plt.style.context('fivethirtyeight'):
#   plt.plot(baseline[:,0], baseline[:,1])
#   plt.plot(baseline[:,0], baseline[:,2])
# plt.show()

## training deeper networks
#
# with plt.style.context('fivethirtyeight'):
#     plt.plot(convA[0:30, 0], convA[0:30, 1], '-', color='red', label = 'ConvA')#,  lw=2,   mew = 2, markersize = 8, markerfacecolor="None", markeredgecolor="orange", label="quad")
#     plt.plot(baseline[0:30,0], baseline[0:30,1], '-', color='blue', label = 'ConvB')#,  lw=2,   mew = 2, markersize = 8, markerfacecolor="None", markeredgecolor="orange", label="quad")
#     plt.plot(convE[0:30, 0], convE[0:30, 1], '-', color='green', label = 'ConvE')#,  lw=2,   mew = 2, markersize = 8, markerfacecolor="None", markeredgecolor="orange", label="quad")
#
# plt.legend(loc = 'upper left')
# plt.title("Comparison of Network Depth")
# plt.ylabel("Train Accuracy")
# plt.xlabel("Number of epochs")
# plt.show()

# batch normalization
# with plt.style.context('fivethirtyeight'):
#     plt.plot(baseline[0:40,0], baseline[0:40,1], '-', color='blue', label = 'Batch Normalization')#,  lw=2,   mew = 2, markersize = 8, markerfacecolor="None", markeredgecolor="orange", label="quad")
#     plt.plot(noBN[0:40, 0], noBN[0:40, 1], '-', color='green', label = 'no Batch Normalization')#,  lw=2,   mew = 2, markersize = 8, markerfacecolor="None", markeredgecolor="orange", label="quad")
#
# plt.legend(loc = 'upper left')
# plt.title("Analysis of Batch Normalization")
# plt.ylabel("Train Accuracy")
# plt.xlabel("Number of epochs")
# plt.show()

#data augmentation

# aug  = pd.read_csv('../data/augment.csv', header = None, quoting=3, sep = ',').as_matrix()
# baseline = pd.read_csv('../data/baseline.csv', header = None, quoting=3, sep = ',').as_matrix()
# with plt.style.context('fivethirtyeight'):
#     plt.plot(baseline[0:12,0], baseline[0:12,1], '-', color='blue', label = 'Baseline')#,  lw=2,   mew = 2, markersize = 8, markerfacecolor="None", markeredgecolor="orange", label="quad")
#     plt.plot(aug[0:12, 0], aug[0:12, 1], '-', color='green', label = 'Data Augmentation')#,  lw=2,   mew = 2, markersize = 8, markerfacecolor="None", markeredgecolor="orange", label="quad")
#
# plt.legend(loc = 'base')
# plt.title("Data Augmentation")
# plt.ylabel("Train Accuracy")
# plt.xlabel("Number of epochs")
# plt.show()

# final Model
fin = pd.read_csv('../data/finalModel.csv', header = None, quoting=3, sep = ',').as_matrix()
baseline = pd.read_csv('../data/baseline.csv', header = None, quoting=3, sep = ',').as_matrix()

with plt.style.context('fivethirtyeight'):
    plt.plot(fin[0:20,0], fin[0:20,2], '-', color='blue', label = 'Post-Hyperparameter Tuning')#,  lw=2,   mew = 2, markersize = 8, markerfacecolor="None", markeredgecolor="orange", label="quad")
    plt.plot(baseline[0:20,0], baseline[0:20, 2], '-', color='green', label = 'Pre-Hyperparameter Tuning')#,  lw=2,   mew = 2, markersize = 8, markerfacecolor="None", markeredgecolor="orange", label="quad")

plt.legend(loc = 'base')
plt.title("Before and After Hyperparameter Tuning")
plt.ylabel("Test Accuracy")
plt.xlabel("Number of epochs")
plt.show()

# single =pd.read_csv('../data/singleLayer_BN_noDropout.csv', header = None, quoting=3, sep=',').as_matrix()
# double =pd.read_csv('../data/doubleLayer_BN_noDropout.csv', header = None, quoting=3, sep=',').as_matrix()
# three =pd.read_csv('../data/threeLayer_BN_noDropout.csv', header = None, quoting=3, sep=',').as_matrix()
#
# with plt.style.context('fivethirtyeight'):
#   plt.plot(single[0:20,0], 100 - single[0:20,1], '--', color='red', label = ' [[CONV - RELU] x 2] - POOL] x 1')
#   plt.plot(double[0:20, 0], 100 - double[0:20,1], '--', color='green', label = ' [[CONV - RELU] x 2] - POOL] x 2 ')
#   plt.plot(three[0:20, 0], 100 - three[0:20, 1], '--', color = 'orange', label = ' [[CONV - RELU] x 2] - POOL] x 3')
# plt.legend(loc = 'best')
# plt.title("Effect of convolutional layers")
# plt.ylabel("Train Error")
# plt.xlabel("Number of epochs")
# plt.show()

# fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 5))
# withBN3 =pd.read_csv('../data/threeLayer_BN_noDropout.csv', header = None, quoting=3, sep=',').as_matrix()
# withoutBN3 =pd.read_csv('../data/threeLayer_noBN_noDropout.csv', header = None, quoting=3, sep=',').as_matrix()
# withBN2 =pd.read_csv('../data/doubleLayer_BN_noDropout.csv', header = None, quoting=3, sep=',').as_matrix()
# withoutBN2 =pd.read_csv('../data/doubleLayer_noBN_noDropout.csv', header = None, quoting=3, sep=',').as_matrix()
# withBN1 =pd.read_csv('../data/singleLayer_BN_noDropout.csv', header = None, quoting=3, sep=',').as_matrix()
# withoutBN1 =pd.read_csv('../data/singleLayer_noBN_noDropout.csv', header = None, quoting=3, sep=',').as_matrix()
# with plt.style.context('fivethirtyeight'):
#   axes[0].plot(withBN1[0:20,0], 100 - withBN1[0:20,1], '--', color='blue', label = 'Batch Norm.')
#   axes[0].plot(withoutBN1[0:20, 0], 100 - withoutBN1[0:20,1], '--', color='red', label = 'No Batch Norm.')
#   axes[1].plot(withBN2[0:20, 0], 100 - withBN2[0:20, 1], '--', color='blue', label='Batch Norm.')
#   axes[1].plot(withoutBN2[0:20, 0], 100 - withoutBN2[0:20, 1], '--', color='red', label='No Batch Norm.')
#   axes[2].plot(withBN3[0:20, 0], 100 - withBN3[0:20, 1], '--', color='blue', label='Batch Norm.')
#   axes[2].plot(withoutBN3[0:20, 0], 100 - withoutBN3[0:20, 1], '--', color='red', label='No Batch Norm.')
# axes[0].legend(loc = 'best')
# axes[1].legend(loc = 'best')
# axes[2].legend(loc = 'best')
# axes[0].set_title("[CONV - RELU] x 2] - POOL] x 1")
# axes[1].set_title("[CONV - RELU] x 2] - POOL] x 2")
# axes[2].set_title("[CONV - RELU] x 2] - POOL] x 3")
# for ax in axes:
#     ax.set_xlabel('Number of Epochs')
#     ax.set_ylabel('Train Error')
# #plt.title("Effect of Interlayer Batch Normalization", y = 1.5)
# plt.setp(axes)
# plt.show()


# fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
# conv3 =pd.read_csv('../data/convA.csv', header = None, quoting=3, sep=',').as_matrix()
# conv5 =pd.read_csv('../data/conv5.csv', header = None, quoting=3, sep=',').as_matrix()
# conv7 =pd.read_csv('../data/conv7.csv', header = None, quoting=3, sep=',').as_matrix()
# with plt.style.context('fivethirtyeight'):
#   axes[0].plot(conv3[0:25,0], 100 - conv3[0:25,1], '--', color='blue', label = '3x3 Filter')
#   axes[0].plot(conv5[0:25, 0], 100 - conv5[0:25,1], '--', color='red', label = '5x5 Filter')
#   axes[0].plot(conv7[0:25, 0], 100 - conv7[0:25, 1], '--', color='orange', label='7x7 Filter')
#   axes[1].plot(conv3[0:25, 0], conv3[0:25, 2], '--', color='blue', label='3x3 Filter')
#   axes[1].plot(conv5[0:25, 0], conv5[0:25, 2], '--', color='red', label='5x5 Filter.')
#   axes[1].plot(conv7[0:25, 0], conv7[0:25, 2], '--', color='orange', label='7x7 Filter.')
# axes[0].legend(loc = 'best')
# axes[1].legend(loc = 'best')
# axes[0].set_title("Training Error")
# axes[1].set_title("Test Accuracy")
# axes[0].set_xlabel('Number of Epochs')
# axes[0].set_ylabel('Train Error')
# axes[1].set_xlabel('Number of Epochs')
# axes[1].set_ylabel('Test Accuracy')
# #plt.title("Effect of Interlayer Batch Normalization", y = 1.5)
# plt.setp(axes)
# plt.show()
