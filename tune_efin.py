import os

import time
import yaml
import random
import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sklift.metrics import uplift_auc_score, qini_auc_score

import optuna
from optuna.samplers import TPESampler

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from utils.io import load_yaml
from utils.helper import uplift_at_k, weighted_average_uplift
from models.architecture import EFIN

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'


class WrapperModel(nn.Module):
    def __init__(self, model):
        super(WrapperModel, self).__init__()
        self.model = model

    def forward(self, feature_list, is_treat, label_list):
        final_output = self.model.calculate_loss(feature_list, is_treat, label_list)
        return final_output


def valid(model, valid_dataloader, device, metric):
    logger.info('Start Verifying')
    model.eval()
    predictions = []
    true_labels = []
    is_treatment = []

    for step, (X, T, valid_label) in enumerate(tqdm(valid_dataloader)):
        model.eval()

        feature_list = X.to(device)
        is_treat = T.to(device)
        label_list = valid_label.to(device)
        _, _, _, _, _, u_tau = model.module.model.forward(feature_list, is_treat)
        uplift = u_tau.squeeze()

        predictions.extend(uplift.detach().cpu().numpy())
        true_labels.extend(label_list.detach().cpu().numpy())
        is_treatment.extend(is_treat.detach().cpu().numpy())

    true_labels = np.array(true_labels)
    predictions = np.array(predictions)
    is_treatment = np.array(is_treatment)

    # Compute uplift at first k observations by uplift of the total sample
    u_at_k = uplift_at_k(true_labels, predictions, is_treatment, strategy='overall', k=0.3)

    # Compute normalized Area Under the Qini curve (aka Qini coefficient) from prediction scores
    qini_coef = qini_auc_score(true_labels, predictions, is_treatment)

    # Compute normalized Area Under the Uplift Curve from prediction scores
    uplift_auc = uplift_auc_score(true_labels, predictions, is_treatment)

    # Weighted average uplift
    wau = weighted_average_uplift(true_labels, predictions, is_treatment, strategy='overall')

    valid_result = [u_at_k, qini_coef, uplift_auc, wau]

    if metric == "AUUC":
        valid_metric = uplift_auc
    elif metric == "QINI":
        valid_metric = qini_coef
    elif metric == 'WAU':
        valid_metric = wau
    else:
        valid_metric = u_at_k
    logger.info("Valid results: {}".format(valid_result))
    return valid_metric, valid_result, true_labels, predictions, is_treatment


class CRITEO(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)

    def load_data(self, data_file):
        data_matrix = np.load(data_file)
        Data = np.float32(data_matrix)
        return Data

    def __len__(self):
        return np.shape(self.data)[0]

    def __getitem__(self, idx):
        return self.data[idx]


def collote_fn(batch_samples):
    batch_input, batch_label, batch_is_treatment = [], [], []
    for sample in batch_samples:
        batch_input.append(sample[:12])
        batch_label.append(sample[13])
        batch_is_treatment.append(sample[12])
    input_list = torch.tensor(batch_input)
    label_list = torch.tensor(batch_label)
    is_treatment_label_list = torch.tensor(batch_is_treatment)

    return input_list, is_treatment_label_list, label_list


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def objective(single_trial):
    trial = optuna.integration.TorchDistributedTrial(single_trial, device=device)

    # sample a set of hyperparameters.
    rank = trial.suggest_categorical('rank', [32, 64, 128])
    rank2 = trial.suggest_categorical('rank2', [32, 64, 128])
    lamb = trial.suggest_categorical('lambda', [1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    batch_size = trial.suggest_categorical('batch_size', [256, 512, 1024, 2048])
    learning_rate = trial.suggest_categorical('learning_rate', [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1])

    # parameter settings
    train_file = "datax/criteo/train.npy"
    valid_file = "datax/criteo/valid.npy"

    setup_seed(seed)

    # model
    model = EFIN(input_dim=12, hc_dim=rank, hu_dim=rank2, is_self=False, act_type="elu")
    model = WrapperModel(model).to(device)

    # data
    train_data = CRITEO(train_file)
    valid_data = CRITEO(valid_file)

    train_sampler = DistributedSampler(train_data, shuffle=True)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size, collate_fn=collote_fn)
    valid_dataloader = DataLoader(valid_data, batch_size=2048, shuffle=False, collate_fn=collote_fn)

    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,
                                                      find_unused_parameters=True)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=lamb)

    best_valid_metric = 0
    break_mask = torch.zeros(1).to(device)
    result_early_stop = 0
    logger.info(f'EFIN: Rank {local_rank} Start Training')
    for epoch in range(num_epoch):
        train_sampler.set_epoch(epoch)
        tr_loss = 0
        tr_steps = 0
        logger.info("Training Epoch: {}/{}".format(epoch + 1, int(num_epoch)))
        for step, (X, T, label) in enumerate(tqdm(train_dataloader)):
            tr_steps += 1

            feature_list = X.to(device)
            is_treat = T.to(device)
            label_list = label.to(device)

            loss = model(feature_list, is_treat, label_list)

            model.train()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tr_loss += loss.item()

        if local_rank == 0:
            logger.info("Epoch loss: {}, Avg loss: {}".format(tr_loss, tr_loss / tr_steps))
            # valid
            model.eval()
            valid_metric, _, _, _, _ = valid(model, valid_dataloader, device, metric)

            if valid_metric > best_valid_metric:
                best_valid_metric = valid_metric
                result_early_stop = 0
            else:
                result_early_stop += 1

                if result_early_stop > 5:
                    break_mask += 1

        dist.barrier()
        dist.all_reduce(break_mask)

        if break_mask == 1:
            break

    return best_valid_metric


# parameter settings
# check
seed = 0
n_trials = 40
num_epoch = 20
metric = 'QINI'

setup_seed(seed)

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
                    level=logging.INFO)
logger = logging.getLogger(__name__)

local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)
device = torch.device('cuda', local_rank)

dist.init_process_group(backend='nccl', init_method='env://')

if local_rank == 0:
    study = optuna.create_study(sampler=TPESampler(seed=seed), direction='maximize')
    study.optimize(objective, n_trials=n_trials)
else:
    for _ in range(n_trials):
        try:
            objective(None)
        except optuna.TrialPruned:
            pass

if local_rank == 0:
    trials = study.trials_dataframe()
    best_params = study.best_params

    table_path = load_yaml('config/global.yml', key='path')['tables']
    if not os.path.exists(table_path + 'criteo/'):
        os.makedirs(table_path + 'criteo/')

    trials.to_csv(table_path + 'criteo/' + 'EFIN-Tune.csv')

    if Path(table_path + 'criteo/' + 'op_hyper_params.yml').exists():
        pass
    else:
        yaml.dump(dict(criteo=dict()),
                  open(table_path + 'criteo/' + 'op_hyper_params.yml', 'w'), default_flow_style=False)
    time.sleep(0.5)
    hyper_params_dict = yaml.safe_load(open(table_path + 'criteo/' + 'op_hyper_params.yml', 'r'))
    hyper_params_dict['criteo']['EFIN'] = best_params
    yaml.dump(hyper_params_dict, open(table_path + 'criteo/' + 'op_hyper_params.yml', 'w'), default_flow_style=False)
