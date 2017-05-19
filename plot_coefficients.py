import matplotlib.pyplot as plt
import pandas as pd

def plot_coefficients(params, ticks_list, title, f_name):
    plt.figure(figsize=(10,15))
    plt.plot(params.median(), range(params.shape[1]), 'ko', ms = 10)
    plt.hlines(range(params.shape[1]), params.quantile(0.025), params.quantile(0.975), 'k')
    plt.hlines(range(params.shape[1]), params.quantile(0.25), params.quantile(0.75), 'k', linewidth = 3)
    plt.axvline(0, linestyle = 'dashed', color = 'k')
    plt.xlabel('Median Coefficient Estimate (50 and 95% CI)')
    plt.yticks(range(params.shape[1]), ticks_list)
    plt.ylim([-1, params.shape[1]])
    plt.xlim([(min(params.quantile(0.025))-0.5), (max(params.quantile(0.975))+0.5)])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f_name)