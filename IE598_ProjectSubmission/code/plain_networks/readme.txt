First download and prepare the dataset using:

	bash download.sh

This generates CIFAR10.hdf5 and its augmented version CIFAR10_augmented.hdf5. We do not use the augmented dataset by default as the paper requires a different augmentation procedure.

Training the resnet:

	python train.py --model models/ResNet.py --batchsize 128 --epoch 170 --lr 0.1 --gpu 0

This should generate a txt file in a results folder that has the training accuracies and test accuracies after every epoch and batch.
