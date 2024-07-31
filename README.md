## KDD23_EFIN

Experiments codes for the paper:

Dugang Liu, Xing Tang, Han Gao, Fuyuan Lyu, Xiuqiang He. Explicit Feature Interaction-aware Uplift Network for Online Marketing. In Proceedings of SIGKDD '23.

**Please cite our SIGKDD '23 paper if you use our codes. Thanks!**


## Requirement

- python==3.8.5
- torch==1.13.1+cu117
- optuna==2.10.0


## Usage

Since the dataset file is too large, please obtain it from the original web page and place it in folder "datax/X". For Criteo dataset, the command line examples are as follows:

**For data processing:**

```bash
python get_criteo.py
```

**For hyperparameter search and training:**

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup python -m torch.distributed.launch --nproc_per_node=8 tune_efin.py > tune_efin 2>&1 &
```

## More info
We have built an initial benchmark for deep uplift modeling, which can be found in this paper ([Link](https://arxiv.org/pdf/2406.00335)). The related project homepage is under construction. Please stay tuned!

## 
If you have any issues or ideas, feel free to contact us ([dugang.ldg@gmail.com](mailto:dugang.ldg@gmail.com)).
