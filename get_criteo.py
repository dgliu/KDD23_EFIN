import os
import argparse
import numpy as np

from utils.io import criteo_statistics
from utils.progress import WorkSplitter
from utils.split import split_seed_randomly
from utils.argcheck import ratio_without_test


def main(args):
    progress = WorkSplitter()

    save_dir = args.path + args.problem
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    progress.section("Criteo: Dataset Statistics")
    data_matrix = criteo_statistics(path=args.path, name=args.problem + args.file, sep=args.sep,
                                    normalize=args.normalize)

    progress.section("Criteo: Split Original Data")
    train_mat, valid_mat = split_seed_randomly(data_matrix=data_matrix, ratio=args.ratio, split_seed=args.seed)
    if args.normalize:
        np.save(args.path + args.problem + 'n_train.npy', train_mat)
        print('Train Size: {0}, {1}'.format(np.shape(train_mat)[0], np.shape(train_mat)[1]))
        np.save(args.path + args.problem + 'n_valid.npy', valid_mat)
        print('Valid Size: {0}, {1}'.format(np.shape(valid_mat)[0], np.shape(train_mat)[1]))
    else:
        np.save(args.path + args.problem + 'train.npy', train_mat)
        print('Train Size: {0}, {1}'.format(np.shape(train_mat)[0], np.shape(train_mat)[1]))
        np.save(args.path + args.problem + 'valid.npy', valid_mat)
        print('Valid Size: {0}, {1}'.format(np.shape(valid_mat)[0], np.shape(train_mat)[1]))

    progress.section("Criteo: Data Preprocess Completed")


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="Data Preprocess")
    parser.add_argument('-p', dest='path', default='datax/')
    parser.add_argument('-d', dest='problem', default='criteo/')
    parser.add_argument('-n', dest='normalize', default=False)
    parser.add_argument('-f', dest='file', help='file name', default='criteo-uplift-v2.1.csv')
    parser.add_argument('-sep', dest='sep', help='separate', default=',')
    parser.add_argument('-s', dest='seed', help='random seed', type=int, default=0)
    parser.add_argument('-r', dest='ratio', type=ratio_without_test, default='0.8,0.2')
    args = parser.parse_args()
    main(args)
