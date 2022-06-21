# -*- coding: utf-8 -*-
# @Author: foxwy
# @Date:   2020-06-02 15:58:09
# @Last Modified by:   WY
# @Last Modified time: 2021-07-29 18:49:15

import os
import sys
import random
import numpy as np
import copy
import matplotlib.pyplot as plt
import scipy
from scipy.linalg import expm
import math
from openpyxl import Workbook
from timeit import default_timer as timer

# -----external libraries-----
sys.path.append('..')

from basic_function import elm
from basic_function import dataset
from basic_function.dataset import RMSE

# ----------function----------
plt.style.use(['science', 'no-latex'])
plt.rcParams["font.family"] = 'Arial'

font_size = 23  # 34, 28, 38
font = {'size': font_size, 'weight': 'normal'}

colors = ['#75bbfd', 'magenta', '#658b38', '#c79fef', '#06c2ac', 'cyan', '#ff9408', '#430541', 'blue', '#fb2943']

def Plt_set(ax, xlabel, ylabel, title, loc=2, ncol=1, f_size=17):
    ax.tick_params(labelsize=font_size)
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['left'].set_linewidth(3)
    ax.spines['right'].set_linewidth(3)
    ax.spines['top'].set_linewidth(3)
    ax.tick_params(width=3)
    font2 = {'size': f_size, 'weight': 'normal'}
    ax.legend(prop=font2, loc=loc, frameon=True, ncol=ncol)
    ax.set_xlabel(xlabel, font)  # fontweight='bold'
    ax.set_ylabel(ylabel, font)
    ax.set_title(title, font)


def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            os.remove(c_path)


def Random_unitary(m):
    U = 1.0
    for i in range(int(math.log(m, 2))):
        theta = random.uniform(0, np.pi)
        U = np.kron(U, np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]))

    return U


def Unitary_cosin(X, m, a):
    U_cos = 1.0
    n = int(math.log(m, 2))
    m_n_int = round(m / n)
    for i in range(n - 1):
        theta = sum(X[i * m_n_int:(i + 1) * m_n_int]**201) / m_n_int * pi / a
        #theta = math.acos(theta)
        U_cos = np.kron(U_cos, np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]))
    theta = sum(X[(n-1) * m_n_int:]**201) / (m - (n - 1) * m_n_int) * pi / a
    #theta = math.acos(theta)
    U_cos = np.kron(U_cos, np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]))

    return U_cos


def Get_H(x, U):
    return 1.0 / (1 + np.exp(-1.0 * x.dot(U)))


def Get_belta(H, y):
    return scipy.linalg.pinv(H.T.dot(H)).dot(H.T.dot(y))


if __name__ == '__main__':
    # ----------define----------
    pi = math.pi
    save_fig_flag = True
    save_excel_flag = True
    show_fig_flag = False
    path = 'figure_final/'

    dataset_names = ['airfoil', 'airquality', 'cbm', 'communities', 'concrete', 'crime', 'demand', 'estate', 'facebook', 'qsar', 'temperature', 'wine']

    for dataset_name in dataset_names:
        if save_excel_flag:
            wb = Workbook()
            ws = wb.active
            ws.append(['RMSE_1', 'RMSE_2', 'RMSE_3', 'RMSE_4',\
                       'Ex_1_time_train', 'Ex_1_time_test', 'Ex_2_time_train', 'Ex_2_time_test',\
                       'Ex_3_time_train', 'Ex_3_time_test', 'Ex_4_time_train', 'Ex_4_time_test'])

        # ----------main process----------
        a = 20

        best_RMSE_3 = np.inf
        best_RMSE_4 = np.inf
        for row in range(1, 101):
            # ----------dataset----------
            #dataset_name = 'communities'

            if 'airfoil' in dataset_name:
                from QELM_regression.figure_final.airfoil import fig
                x, x_test, y, y_test = dataset.load_airfoil_dataset()
            if 'airquality' in dataset_name:
                from QELM_regression.figure_final.airquality import fig
                x, x_test, y, y_test = dataset.load_airquality_dataset()
            elif 'cbm' in dataset_name:
                from QELM_regression.figure_final.cbm import fig
                x, x_test, y, y_test = dataset.load_cbm_dataset()
            elif 'communities' in dataset_name:
                from QELM_regression.figure_final.communities import fig
                x, x_test, y, y_test = dataset.load_communities_dataset(1)
            elif 'concrete' in dataset_name:
                from QELM_regression.figure_final.concrete import fig
                x, x_test, y, y_test = dataset.load_concrete_dataset(1)
            elif 'crime' in dataset_name:
                from QELM_regression.figure_final.crime import fig
                x, x_test, y, y_test = dataset.load_crime_dataset(2, 129)
            elif 'demand' in dataset_name:
                from QELM_regression.figure_final.demand import fig
                x, x_test, y, y_test = dataset.load_demand_dataset()
            elif 'estate' in dataset_name:
                from QELM_regression.figure_final.estate import fig
                x, x_test, y, y_test = dataset.load_estate_dataset()
            elif 'facebook' in dataset_name:
                from QELM_regression.figure_final.facebook import fig
                x, x_test, y, y_test = dataset.load_facebook_dataset()
            elif 'qsar' in dataset_name:
                from QELM_regression.figure_final.qsar import fig
                x, x_test, y, y_test = dataset.load_qsar_dataset()
            elif 'temperature' in dataset_name:
                from QELM_regression.figure_final.temperature import fig
                x, x_test, y, y_test = dataset.load_temperature_dataset()
            elif 'wine' in dataset_name:
                from QELM_regression.figure_final.wine import fig
                x, x_test, y, y_test = dataset.load_wine_dataset()

            N_train = len(x)
            N_test = len(x_test)
            m = len(x[0])
            print('\ndataset, N_train, N_test, m, row: ', dataset_name, N_train, N_test, m, row)

            # ----------unitary matrix-nonlinear----------
            # ---train
            Unitary_cosin_all = []
            for x_i in x:
                Unitary_cosin_all.append(Unitary_cosin(x_i, m, a))

            # ---test
            Unitary_cosin_test = []
            for x_test_i in x_test:
                Unitary_cosin_test.append(Unitary_cosin(x_test_i, m, a))


            plt.close('all')
            plt.figure(figsize=(12, 8))


            # ----------ELM----------
            # -----ELM train-----
            time_begin = timer()
            P, A, beta = elm.ELM_train(x, y, 1000, m)  # C, L
            time_end = timer()
            time_exp1_train = time_end - time_begin

            # -----elm test-----
            time_begin = timer()
            y_predict = elm.ELM_predict(x_test, A, beta)
            time_end = timer()
            time_exp1_test = time_end - time_begin

            RMSE_1 = RMSE(y_test, y_predict)
            print('ELM train time, test time, test RMSE:', time_exp1_train, time_exp1_test, RMSE_1)

            ax = plt.subplot(221)
            ax.plot([i for i in range(N_test)], y_test, label='Real', color=colors[0], linewidth=3)
            ax.plot([i for i in range(N_test)], y_predict, label='Predict', color=colors[6], linewidth=3)
            Plt_set(ax, 'Test sample', 'Output value', 'Ex_1')


            # ----------ELM+U_random simulate (H=XW, H\beta=Y)----------
            time_begin = timer()
            # -----random unitary-----
            U = Random_unitary(m)

            # -----train-----
            H = Get_H(x, U)
            beta = Get_belta(H, y)
            time_end = timer()
            time_exp2_train = time_end - time_begin

            # -----test-----
            time_begin = timer()
            H_test = Get_H(x_test, U)
            Qelm_y_predict = H_test.dot(beta)
            time_end = timer()
            time_exp2_test = time_end - time_begin

            # -----result, figure-----
            RMSE_2 = RMSE(y_test, Qelm_y_predict)
            print('ELM+U_random train time, test time, test RMSE:', time_exp2_train, time_exp2_test, RMSE_2)

            ax = plt.subplot(222)
            plt.plot([i for i in range(N_test)], y_test, label='Real', color=colors[0], linewidth=3)
            plt.plot([i for i in range(N_test)], Qelm_y_predict, label='Predict', color=colors[6], linewidth=3)
            Plt_set(ax, 'Test sample', 'Output value', 'Ex_2')


            # ----------Quantum ELM+U_random simulate (H=XW, H\beta=Y)----------
            # -----train-----
            time_begin = timer()
            B_train = x.dot(U)
            #H = np.sum(B_train*Unitary_cosin_all, axis=2)
            H = np.zeros_like(B_train)
            for i in range(N_train):
                H[i] = B_train[i].dot(Unitary_cosin_all[i])
            beta = Get_belta(H, y)
            time_end = timer()
            time_exp3_train = time_end - time_begin

            # -----test-----
            time_begin = timer()
            B_test = x_test.dot(U)
            #H_test = np.sum(B_test*Unitary_cosin_test, axis=2)
            H_test = np.zeros_like(B_test)
            for i in range(N_test):
                H_test[i] = B_test[i].dot(Unitary_cosin_test[i])

            Qelm_y_predict = H_test.dot(beta)
            time_end = timer()
            time_exp3_test = time_end - time_begin

            # -----result, figure-----
            RMSE_3 = RMSE(y_test, Qelm_y_predict)
            print('QELM+U_random train time, test time, test RMSE:', time_exp3_train, time_exp3_test, RMSE_3)

            ax = plt.subplot(223)
            plt.plot([i for i in range(N_test)], y_test, label='Real', color=colors[0], linewidth=3)
            plt.plot([i for i in range(N_test)], Qelm_y_predict, label='Predict', color=colors[6], linewidth=3)
            Plt_set(ax, 'Test sample', 'Output value', 'Ex_3')


            # ----------Quantum ELM simulate (H=XW, H\beta=Y)----------
            # -----train-----
            time_begin = timer()
            #H = np.sum(x[:,np.newaxis,:]*Unitary_cosin_all, axis=2)
            H = np.zeros_like(x)
            for i in range(N_train):
                H[i] = x[i].dot(Unitary_cosin_all[i])
            beta = Get_belta(H, y)
            time_end = timer()
            time_exp4_train = time_end - time_begin

            # -----test-----
            time_begin = timer()
            #H_test = np.sum(x_test[:,np.newaxis,:]*Unitary_cosin_test, axis=2)
            H_test = np.zeros_like(x_test)
            for i in range(N_test):
                H_test[i] = x_test[i].dot(Unitary_cosin_test[i])

            Qelm_y_predict = H_test.dot(beta)
            time_end = timer()
            time_exp4_test = time_end - time_begin

            # -----result, figure-----
            RMSE_4 = RMSE(y_test, Qelm_y_predict)
            print('QELM train time, test time, test RMSE:', time_exp4_train, time_exp4_test, RMSE_4)

            ax = plt.subplot(224)
            plt.plot([i for i in range(N_test)], y_test, label='Real', color=colors[0], linewidth=3)
            plt.plot([i for i in range(N_test)], Qelm_y_predict, label='Predict', color=colors[6], linewidth=3)
            Plt_set(ax, 'Test sample', 'Output value', 'Ex_4')


            if RMSE_3 < best_RMSE_3 and RMSE_4 < best_RMSE_4:
                best_RMSE_3 = RMSE_3
                best_RMSE_4 = RMSE_4
                # ----------save-----------
                # -----fig------
                if save_fig_flag:
                    fig_path = path + dataset_name + '/fig/' + str(RMSE_1) + '_' + str(RMSE_2) + '_' + str(RMSE_3) + '_' + str(RMSE_4)
                    if row == 1:
                        del_file(path + dataset_name + '/fig')
                    plt.tight_layout()
                    plt.savefig(fig_path + '.pdf', dpi=600, bbox_inches='tight', pad_inches=0.05)

                if show_fig_flag:
                    plt.show()

            # -----excel-----
            if save_excel_flag:
                ws.append([RMSE_1, RMSE_2, RMSE_3, RMSE_4, time_exp1_train, time_exp1_test, \
                           time_exp2_train, time_exp2_test, time_exp3_train, time_exp3_test, time_exp4_train, time_exp4_test])
        if save_excel_flag:
            wb.save(path + dataset_name + '/results.xlsx')
            fig.Plot_data()
