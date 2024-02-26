import yaml
import random
import numpy as np
import pandas as pd


def ali_sub_cleaning(path, part, data, feature1, feature2, sep):
    random.seed(0)
    np.random.seed(0)

    sample_size = 0
    cleaned_data = []

    k1 = pd.read_csv(path + part + feature1, sep=sep, header=0)
    d1 = k1.set_index('key1').agg(list, 1).to_dict()
    print('key feature1 load')

    k2 = pd.read_csv(path + part + feature2, sep=sep, header=0)
    d2 = k2.set_index('key2').agg(list, 1).to_dict()
    print('key feature2 load')

    with open(path + part + data, 'r', encoding='utf-8') as f:
        for line in f:
            if 'sample_id' in line:
                continue

            _line = line.split(sep)
            # Remove samples whose features are not in key1 and key2
            if (int(_line[1]) not in d1) or (int(_line[2]) not in d2):
                sample_size += 1
                if sample_size % 1000000 == 0:
                    print('{0} have processed'.format(sample_size))
                continue

            sample_size += 1
            if sample_size % 1000000 == 0:
                print('{0} have processed'.format(sample_size))

            cleaned_data.append([_line[3], _line[10], _line[5], _line[6], _line[7], _line[8], _line[1], _line[2]])

    cleaned_data = np.array(cleaned_data)
    index = np.arange(np.size(cleaned_data, 0))
    np.random.shuffle(index)
    sub_data = cleaned_data[:int(0.01 * len(index)), :]
    np.savetxt(path + part + 'sub_' + data, sub_data, delimiter=';', fmt='%s')
    print('sub size', np.size(sub_data, 0))
    return np.size(sub_data, 0)


def statistics_vlength_features(path, part, data, feature1, feature2, sep):
    log_data = pd.read_csv(path + part + 'sub_' + data, sep=sep,
                           names=['label', 't', 'f1', 'f2', 'f3', 'f4', 'key1', 'key2'], header=None)
    key1_set = set(log_data['key1'])
    key2_set = set(log_data['key2'])
    print('key1 num {0}'.format(len(key1_set)))
    print('key2 num {0}'.format(len(key2_set)))

    key1_vl_feature_len = [0, 0, 0, 0, 0, 0, 0, 0]
    key1_vl_feature_set = [set(), set(), set(), set(), set(), set(), set(), set()]
    with open(path + part + feature1, 'r', encoding='utf-8') as f:
        for line in f:
            if 'key1' in line:
                continue

            _line = line.split(sep)
            if int(_line[0]) in key1_set:
                for i in range(1, 9):
                    if (not pd.isnull(_line[i])) and _line[i] != '':
                        l = _line[i].split(',')

                        if len(l) > key1_vl_feature_len[i-1]:
                            key1_vl_feature_len[i-1] = len(l)

                        key1_vl_feature_set[i-1].update(l)
    f.close()
    key1_vl_feature_max = list(map(len, key1_vl_feature_set))

    key2_vl_feature_len = 0
    key2_vl_feature_set = set()
    with open(path + part + feature2, 'r', encoding='utf-8') as f:
        for line in f:
            if 'key2' in line:
                continue

            _line = line.split(sep)
            if int(_line[0]) in key2_set:
                if (not pd.isnull(_line[1])) and _line[1] != '':
                    l = _line[1].split(',')

                    if len(l) > key2_vl_feature_len:
                        key2_vl_feature_len = len(l)

                    key2_vl_feature_set.update(l)
    f.close()
    key2_vl_feature_max = len(key2_vl_feature_set)
    print('key1_v1_feature_len', key1_vl_feature_len)
    print('key1_v1_feature_max', key1_vl_feature_max)
    print('key2_v1_feature_len', key2_vl_feature_len)
    print('key2_v1_feature_max', key2_vl_feature_max)

    return key1_vl_feature_len, key1_vl_feature_max, key2_vl_feature_len, key2_vl_feature_max


def reindex_all_features(path, part, data, feature1, feature2, sep, key1_len, key1_max, key2_len, key2_max, sub_size):
    k1 = pd.read_csv(path + part + feature1, sep=sep, header=0)
    d1 = k1.set_index('key1').agg(list, 1).to_dict()
    print('key feature1 load')

    k2 = pd.read_csv(path + part + feature2, sep=sep, header=0)
    d2 = k2.set_index('key2').agg(list, 1).to_dict()
    print('key feature2 load')

    sample_size, nums_treatment, nums_control, visit_treatment, visit_control = 0, 0, 0, 0, 0

    # log_feature_num = 4
    log_feature_idx, log_feature_dict = [0, 0, 0, 0], [dict(), dict(), dict(), dict()]

    # key1_feature_num = 13
    key1_feature_idx, key1_feature_dict = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [
        dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict()]

    # key2_feature_num = 8
    key2_feature_idx, key2_feature_dict = [0, 0, 0, 0, 0, 0, 0, 0], [dict(), dict(), dict(), dict(), dict(), dict(),
                                                                     dict(), dict()]

    reindex_data = []
    with open(path + part + 'sub_' + data, 'r', encoding='utf-8') as f:
        for line in f:
            reindex_line = []
            if 'sample_id' in line:
                continue

            _line = line.split(sep)

            sample_size += 1
            if sample_size % 1000000 == 0:
                print('{0} have processed'.format(sample_size))

            if _line[1] == '0':
                nums_control += 1
                if _line[0] == '1':
                    visit_control += 1
            else:
                nums_treatment += 1
                if _line[0] == '1':
                    visit_treatment += 1
            reindex_line.append(_line[0])
            reindex_line.append(_line[1])

            if _line[2] not in log_feature_dict[0]:
                log_feature_dict[0][_line[2]] = str(log_feature_idx[0])
                log_feature_idx[0] += 1
            if _line[3] not in log_feature_dict[1]:
                log_feature_dict[1][_line[3]] = str(log_feature_idx[1])
                log_feature_idx[1] += 1
            if _line[4] not in log_feature_dict[2]:
                log_feature_dict[2][_line[4]] = str(log_feature_idx[2])
                log_feature_idx[2] += 1
            if _line[5] not in log_feature_dict[3]:
                log_feature_dict[3][_line[5]] = str(log_feature_idx[3])
                log_feature_idx[3] += 1
            reindex_line.append(log_feature_dict[0][_line[2]])
            reindex_line.append(log_feature_dict[1][_line[3]])
            reindex_line.append(log_feature_dict[2][_line[4]])
            reindex_line.append(log_feature_dict[3][_line[5]])

            key = d1[int(_line[6])]
            for i in range(8):
                if not pd.isnull(key[i]):
                    l = key[i].replace(':1.0', '').split(',')
                    for j in range(len(l)):
                        if l[j] not in key1_feature_dict[i]:
                            key1_feature_dict[i][l[j]] = str(key1_feature_idx[i])
                            key1_feature_idx[i] += 1
                        reindex_line.append(key1_feature_dict[i][l[j]])

                    reindex_line.extend([str(key1_max[i])] * (key1_len[i] - len(l)))
                else:
                    reindex_line.extend([str(key1_max[i])] * key1_len[i])
            for i in range(8, 13):
                if key[i] not in key1_feature_dict[i]:
                    key1_feature_dict[i][key[i]] = str(key1_feature_idx[i])
                    key1_feature_idx[i] += 1
                reindex_line.append(key1_feature_dict[i][key[i]])

            key = d2[int(_line[7])]
            if not pd.isnull(key[0]):
                l = key[0].replace(':1.0', '').split(',')
                for j in range(len(l)):
                    if l[j] not in key2_feature_dict[0]:
                        key2_feature_dict[0][l[j]] = str(key2_feature_idx[0])
                        key2_feature_idx[0] += 1
                    reindex_line.append(key2_feature_dict[0][l[j]])

                reindex_line.extend([str(key2_max)] * (key2_len - len(l)))
            else:
                reindex_line.extend([str(key2_max)] * key2_len)
            for i in range(1, 8):
                if key[i] not in key2_feature_dict[i]:
                    key2_feature_dict[i][key[i]] = str(key2_feature_idx[i])
                    key2_feature_idx[i] += 1
                reindex_line.append(key2_feature_dict[i][key[i]])

            reindex_data.append(reindex_line)

    print('Size: {0}'.format(sample_size))
    print('Ratio of Treatment to Control: {0}'.format(nums_treatment / nums_control))
    print('Average Visit Ratio: {0}'.format((visit_control + visit_treatment) / sample_size))
    uplift_treatment = visit_treatment / nums_treatment
    uplift_control = visit_control / nums_control
    print('Relative Average Uplift: {0}'.format((uplift_treatment - uplift_control) / uplift_control))
    print('Average Uplift: {0}'.format(uplift_treatment - uplift_control))

    with open(path + part + 'train_' + data, 'w') as f:
        for line in reindex_data[:int(0.8 * sub_size)]:
            f.write(';'.join(line) + '\n')
    f.close()

    with open(path + part + 'valid_' + data, 'w') as f:
        for line in reindex_data[int(0.8 * sub_size):]:
            f.write(';'.join(line) + '\n')
    f.close()

    print('log_feature_idx', log_feature_idx)
    print('key1_feature_idx', key1_feature_idx)
    print('key2_feature_idx', key2_feature_idx)


def criteo_statistics(path, name, sep, normalize):
    raw_data = pd.read_csv(path + name, sep=sep, header=0)
    data_matrix = raw_data.to_numpy()

    sample_size = np.shape(data_matrix)[0]
    print('Original Size: {0}'.format(sample_size))
    nums_treatment = np.sum(data_matrix[:, 12] == 1)
    nums_control = sample_size - nums_treatment
    print('Ratio of Treatment to Control: {0}'.format(nums_treatment/nums_control))
    visit_ratio = np.sum(data_matrix[:, 14] == 1) / sample_size
    print('Average Visit Ratio: {0}'.format(visit_ratio))
    uplift_treatment = np.sum((data_matrix[:, 12] == 1) & (data_matrix[:, 14] == 1)) / nums_treatment
    uplift_control = np.sum((data_matrix[:, 12] == 0) & (data_matrix[:, 14] == 1)) / nums_control
    print('Relative Average Uplift: {0}'.format((uplift_treatment - uplift_control) / uplift_control))
    print('Average Uplift: {0}'.format(uplift_treatment - uplift_control))

    data_matrix = raw_data.drop_duplicates().to_numpy()
    data_matrix = data_matrix[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14]]
    if normalize:
        data_matrix[:, :12] = (data_matrix[:, :12] - np.min(data_matrix[:, :12])) / (
                np.max(data_matrix[:, :12]) - np.min(data_matrix[:, :12]))
    print('Current Size: {0}'.format(np.shape(data_matrix)[0]))
    return data_matrix


def load_yaml(path, key='parameters'):
    with open(path, 'r') as stream:
        try:
            return yaml.safe_load(stream)[key]
        except yaml.YAMLError as exc:
            print(exc)
