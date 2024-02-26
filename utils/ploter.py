import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklift.metrics import qini_curve, perfect_qini_curve, uplift_curve, perfect_uplift_curve, treatment_balance_curve
from utils.helper import uplift_by_percentile

font = {'weight': 'bold',
        'size': 15}

matplotlib.rc('font', **font)


def plot_qini_curve(y_true, uplift, treatment,
                    random=True, perfect=True, negative_effect=True, name=None, fill=False, **kwargs):
    x_actual, y_actual = qini_curve(y_true, uplift, treatment)

    if random:
        x_baseline, y_baseline = x_actual, x_actual * y_actual[-1] / len(y_true)
    else:
        x_baseline, y_baseline = None, None

    if perfect:
        x_perfect, y_perfect = perfect_qini_curve(y_true, treatment, negative_effect)
    else:
        x_perfect, y_perfect = None, None

    fig, ax = plt.subplots()
    p1 = plt.plot(x_actual, y_actual, linewidth=2.5, linestyle='-', label=name)

    if random:
        p2 = plt.plot(x_baseline, y_baseline, linewidth=2.5, linestyle='-', label="Random")

        if fill:
            plt.fill_between(x_actual, y_actual, y_baseline, alpha=0.2)

    if perfect:
        p3 = plt.plot(x_perfect, y_perfect, label="Perfect")

        plt.legend((p1[0], p2[0], p3[0]),
                   (name, 'Random', 'Perfect'),
                   loc='lower left', bbox_to_anchor=(0, 1.02, 1, 0.2), ncol=3, mode="expand", borderaxespad=0,
                   handlelength=1.8, labelspacing=0.0, handletextpad=0.2)
    else:
        plt.legend((p1[0], p2[0]),
                   (name, 'Random'),
                   loc='lower left', bbox_to_anchor=(0, 1.02, 1, 0.2), ncol=2, mode="expand", borderaxespad=0,
                   handlelength=1.8, labelspacing=0.0, handletextpad=0.2)

    ax.set_xlabel('Number targeted', color='black', fontsize=15, fontweight='bold')
    ax.set_ylabel('Number of incremental outcome', color='black', fontsize=15, fontweight='bold')

    plt.tight_layout()
    plt.gca().yaxis.grid(True)

    return plt


def plot_uplift_curve(y_true, uplift, treatment,
                      random=True, perfect=True, name=None, fill=False, **kwargs):
    x_actual, y_actual = uplift_curve(y_true, uplift, treatment)

    if random:
        x_baseline, y_baseline = x_actual, x_actual * y_actual[-1] / len(y_true)
    else:
        x_baseline, y_baseline = None, None

    if perfect:
        x_perfect, y_perfect = perfect_uplift_curve(y_true, treatment)
    else:
        x_perfect, y_perfect = None, None

    fig, ax = plt.subplots()
    p1 = plt.plot(x_actual, y_actual, linewidth=2.5, linestyle='-', label=name)

    if random:
        p2 = plt.plot(x_baseline, y_baseline, linewidth=2.5, linestyle='-', label="Random")

        if fill:
            plt.fill_between(x_actual, y_actual, y_baseline, alpha=0.2)

    if perfect:
        p3 = plt.plot(x_perfect, y_perfect, label="Perfect")

        plt.legend((p1[0], p2[0], p3[0]),
                   (name, 'Random', 'Perfect'),
                   loc='lower left', bbox_to_anchor=(0, 1.02, 1, 0.2), ncol=3, mode="expand", borderaxespad=0,
                   handlelength=1.8, labelspacing=0.0, handletextpad=0.2)
    else:
        plt.legend((p1[0], p2[0]),
                   (name, 'Random'),
                   loc='lower left', bbox_to_anchor=(0, 1.02, 1, 0.2), ncol=2, mode="expand", borderaxespad=0,
                   handlelength=1.8, labelspacing=0.0, handletextpad=0.2)

    ax.set_xlabel('Number targeted', color='black', fontsize=15, fontweight='bold')
    ax.set_ylabel('Number of incremental outcome', color='black', fontsize=15, fontweight='bold')

    plt.tight_layout()
    plt.gca().yaxis.grid(True)

    return plt


def plot_uplift_by_percentile(y_true, uplift, treatment, strategy='overall', kind='line', bins=10,
                              string_percentiles=True, fill=False):
    df = uplift_by_percentile(y_true, uplift, treatment, strategy=strategy,
                              std=True, total=True, bins=bins, string_percentiles=False)

    percentiles = df.index[:bins].values.astype(float)

    response_rate_trmnt = df.loc[percentiles, 'response_rate_treatment'].values
    std_trmnt = df.loc[percentiles, 'std_treatment'].values

    response_rate_ctrl = df.loc[percentiles, 'response_rate_control'].values
    std_ctrl = df.loc[percentiles, 'std_control'].values

    uplift_score = df.loc[percentiles, 'uplift'].values
    std_uplift = df.loc[percentiles, 'std_uplift'].values

    uplift_weighted_avg = df.loc['total', 'uplift']

    if kind == 'line':
        _, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 6))
        plt.errorbar(percentiles, response_rate_trmnt, yerr=std_trmnt,
                     linewidth=2, color='forestgreen', label='treatment\nresponse rate')
        plt.errorbar(percentiles, response_rate_ctrl, yerr=std_ctrl,
                     linewidth=2, color='orange', label='control\nresponse rate')
        plt.errorbar(percentiles, uplift_score, yerr=std_uplift,
                     linewidth=2, color='red', label='uplift')
        if fill:
            plt.fill_between(percentiles, response_rate_trmnt, response_rate_ctrl, alpha=0.1, color='red')

        if np.amin(uplift_score) < 0:
            ax.axhline(y=0, color='black', linewidth=1)

        if string_percentiles:  # string percentiles for plotting
            percentiles_str = [f"0-{percentiles[0]:.0f}"] + \
                              [f"{percentiles[i]:.0f}-{percentiles[i + 1]:.0f}" for i in range(len(percentiles) - 1)]
            ax.set_xticks(percentiles)
            ax.set_xticklabels(percentiles_str, rotation=45)
        else:
            ax.set_xticks(percentiles)

        ax.legend(loc='upper right')
        ax.set_xlabel('Percentile')
        ax.set_ylabel(
            'Uplift = treatment response rate - control response rate')

    else:  # kind == 'bar'
        delta = percentiles[0]
        fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(8, 6), sharex=True, sharey=True)
        fig.text(0.04, 0.5, 'Uplift = treatment response rate - control response rate',
                 va='center', ha='center', rotation='vertical')

        ax[1].bar(np.array(percentiles) - delta / 6, response_rate_trmnt, delta / 3,
                    yerr=std_trmnt, color='forestgreen', label='treatment\nresponse rate')
        ax[1].bar(np.array(percentiles) + delta / 6, response_rate_ctrl, delta / 3,
                    yerr=std_ctrl, color='orange', label='control\nresponse rate')
        ax[0].bar(np.array(percentiles), uplift_score, delta / 1.5,
                    yerr=std_uplift, color='red', label='uplift')

        ax[0].legend(loc='upper right')
        ax[0].tick_params(axis='x', bottom=False)
        ax[0].axhline(y=0, color='black', linewidth=1)
        ax[0].set_title(
            f'Uplift by percentile\nweighted average uplift = {uplift_weighted_avg:.4f}')

        if string_percentiles:  # string percentiles for plotting
            percentiles_str = [f"0-{percentiles[0]:.0f}"] + \
                              [f"{percentiles[i]:.0f}-{percentiles[i + 1]:.0f}" for i in range(len(percentiles) - 1)]
            ax[1].set_xticks(percentiles)
            ax[1].set_xticklabels(percentiles_str, rotation=45)

        else:
            ax[1].set_xticks(percentiles)

        ax[1].legend(loc='upper right')
        ax[1].axhline(y=0, color='black', linewidth=1)
        ax[1].set_xlabel('Percentile')
        ax[1].set_title('Response rate by percentile')

    return plt


def plot_treatment_balance_curve(uplift, treatment, random=True, name=None, winsize=0.1, fill=False):
    x_tb, y_tb = treatment_balance_curve(
        uplift, treatment, winsize=int(len(uplift) * winsize))

    _, ax = plt.subplots(ncols=1, nrows=1, figsize=(14, 7))

    plt.plot(x_tb, y_tb, label=name, color='b')

    if random:
        y_tb_random = np.average(treatment) * np.ones_like(x_tb)

        plt.plot(x_tb, y_tb_random, label='Random', color='black')
        if fill:
            plt.fill_between(x_tb, y_tb, y_tb_random, alpha=0.2, color='b')

    ax.legend()
    ax.set_xlabel('Percentage targeted')
    ax.set_ylabel('Balance: treatment / (treatment + control)')

    return plt
