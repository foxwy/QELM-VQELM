# -*- coding: utf-8 -*-
# @Author: foxwy
# @Date:   2020-06-04 20:41:58
# @Last Modified by:   WY
# @Last Modified time: 2021-10-30 16:40:41

import random
import math
import matplotlib.pyplot as plt

plt.style.use(['science', 'no-latex'])
plt.rcParams["font.family"] = 'Arial'

font_size = 38  # 34, 28, 38
font = {'size': font_size, 'weight': 'normal'}

colors = ['#75bbfd', 'magenta', '#658b38', '#c79fef', '#06c2ac', 'cyan', '#ff9408', '#430541', 'blue', '#fb2943']

def Plt_set(ax, xlabel, ylabel, savepath, log_flag=0, loc=2, ncol=1, f_size=34):
    ax.tick_params(labelsize=font_size)
    ax.spines['bottom'].set_linewidth(6)
    ax.spines['left'].set_linewidth(6)
    ax.spines['right'].set_linewidth(6)
    ax.spines['top'].set_linewidth(6)
    ax.tick_params(width=6)
    font2 = {'size': f_size, 'weight': 'normal'}
    ax.legend(prop=font2, loc=loc, frameon=True, ncol=ncol)
    ax.set_xlabel(xlabel, font)  # fontweight='bold'
    ax.set_ylabel(ylabel, font)

    if log_flag == 1:
        ax.set_xscale('log')
    if log_flag == 2:
        ax.set_yscale('log')
    if log_flag == 3:
        ax.set_xscale('log')
        ax.set_yscale('log')

    plt.savefig(savepath + '.pdf', dpi=600, bbox_inches='tight', pad_inches=0.05)


if __name__ == '__main__':
    pi = math.pi
    for a in [5, 20]:
        for b in [11, 51, 201]:

            x1 = [random.uniform(-1, 1) for _ in range(1000)]
            x2 = [(random.randint(0, 1) * 2 - 1) * math.sqrt(1 - i**2) for i in x1]

            y1 = []
            y2 = []

            for i in range(len(x1)):
                alpha = (x1[i]**b + x2[i]**b) / 2 * pi / a
                #alpha = math.acos(alpha)
                #print(alpha)
                y1.append(x1[i] * math.cos(alpha) - x2[i] * math.sin(alpha))
                y2.append(x1[i] * math.sin(alpha) + x2[i] * math.cos(alpha))

            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.scatter(x1, y1, label='$\\rm y_1$-$\\rm x_1$', s=10, color=colors[0], linewidth=10)
            ax.scatter(x2, y2, label='$\\rm y_2$-$\\rm x_2$', s=10, color=colors[6], linewidth=10)

            ax.set_xticks([-1, -0.5, 0, 0.5, 1])
            ax.set_yticks([-1, -0.5, 0, 0.5, 1])
            Plt_set(ax, '$x$', '$y$', 'fig_activation/a'+str(a)+'_b'+str(b))
            plt.show()
