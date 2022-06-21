# -*- coding: utf-8 -*-
import sys
import random
import math
import numpy as np
import pandas as pd
from numpy.linalg import norm
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder, normalize
from sklearn.model_selection import train_test_split
import urllib


# RMSE
def RMSE(y_real, y_pre):
    return np.sqrt(metrics.mean_squared_error(y_real, y_pre))

def ACC(y_real, y_pre):
    y_real = np.argmax(y_real, axis=1)
    y_pre = np.argmax(y_pre, axis=1)
    acc = metrics.precision_score(y_real, y_pre, average='micro')

    return acc


# ---------------------------------------dataset---------------------------------------
# if log(m_init) is not a integer, add partical original data to dataset
def Add_data(x):
    m_init = len(x[0])
    if math.pow(2, int(math.log(m_init, 2))) != m_init:
        m = int(math.pow(2, int(math.log(m_init, 2)) + 1))
        m_insert = m - m_init
        ins_zeros = np.zeros((len(x), m_insert))
        x = np.insert(x, m_init, ins_zeros.T, 1)

    return x

def Dataset_split(X, y, cls_f=0):
    one_add = np.ones((len(X), 1))
    X = np.insert(X, len(X[0]), one_add.T, 1)
    X = normalize(Add_data(X))
    if cls_f == 0:  # regression
        y = y / norm(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    return X_train, X_test, y_train, y_test

# --------------------classification dataset--------------------
# https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Original%29
def load_breastcarcer_dataset():
    breast = pd.read_table('../dataset/breast-cancer-wisconsin.csv', sep=',', header=None)
    columns_x = list(range(9))
    X = breast.loc[:, columns_x]
    y = breast.loc[:, 9]
    X = np.array(X)
    y = np.array(y)
    en_one = OneHotEncoder()
    T = en_one.fit_transform(y.reshape(-1, 1)).toarray()

    return Dataset_split(X, T, 1)

# https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Coimbra
def load_breastcarcer_coimbra_dataset():
    breast = pd.read_table('../dataset/breast-cancer-coimbra.csv', sep=',', header=None)
    columns_x = list(range(9))
    X = breast.loc[:, columns_x]
    y = breast.loc[:, 9]
    X = np.array(X)
    y = np.array(y)
    en_one = OneHotEncoder()
    T = en_one.fit_transform(y.reshape(-1, 1)).toarray()

    return Dataset_split(X, T, 1)

# https://archive.ics.uci.edu/ml/datasets/Iranian+Churn+Dataset
def load_churn_dataset():
    churn = pd.read_table('../dataset/Churn_Dateset.csv', sep=',', header=None)
    columns_x = list(range(11))
    X = churn.loc[:, columns_x]
    y = churn.loc[:, 11]
    X = np.array(X)
    y = np.array(y)
    en_one = OneHotEncoder()
    T = en_one.fit_transform(y.reshape(-1, 1)).toarray()

    return Dataset_split(X, T, 1)

# https://archive.ics.uci.edu/ml/datasets/Electrical+Grid+Stability+Simulated+Data+
def load_electrical_grid_dataset():
    grid = pd.read_table('../dataset/electrical_grid.csv', sep=',', header=None)
    columns_x = list(range(13))
    X = grid.loc[:, columns_x]
    y = grid.loc[:, 13]
    X = np.array(X)
    y = np.array(y)
    en_one = OneHotEncoder()
    T = en_one.fit_transform(y.reshape(-1, 1)).toarray()

    return Dataset_split(X, T, 1)

# https://archive.ics.uci.edu/ml/datasets/Algerian+Forest+Fires+Dataset++
def load_forestfires_dataset():
    fires = pd.read_table('../dataset/Algerian_forest_fires.csv', sep=',', header=None)
    #print(fires)
    columns_x = list(range(10))
    X = fires.loc[:, columns_x]
    y = fires.loc[:, 10]
    X = np.array(X)
    y = np.array(y)
    en_one = OneHotEncoder()
    T = en_one.fit_transform(y.reshape(-1, 1)).toarray()

    return Dataset_split(X, T, 1)

# https://archive.ics.uci.edu/ml/datasets/Haberman%27s+Survival
def load_haberman_dataset():
    haberman = pd.read_table('../dataset/haberman.csv', sep=',', header=None)
    columns_x = list(range(3))
    X = haberman.loc[:, columns_x]
    y = haberman.loc[:, 3]
    X = np.array(X)
    y = np.array(y)
    en_one = OneHotEncoder()
    T = en_one.fit_transform(y.reshape(-1, 1)).toarray()

    return Dataset_split(X, T, 1)

# https://archive.ics.uci.edu/ml/datasets/HCC+Survival
def load_hcc_dataset():
    comm = pd.read_table('../dataset/hcc-data.csv', sep=',', na_values='?', header=None)
    columns_x = list(range(33))
    X = comm.loc[:, columns_x]
    X = X.fillna(0)
    y = comm.loc[:, 33]
    y = y.fillna(0)
    X = np.array(X)
    y = np.array(y)
    en_one = OneHotEncoder()
    T = en_one.fit_transform(y.reshape(-1, 1)).toarray()

    return Dataset_split(X, T, 1)

# https://archive.ics.uci.edu/ml/datasets/Madelon
def load_madelon_dataset():
    madelon = pd.read_table('../dataset/madelon.csv', sep=',', header=None)
    columns_x = list(range(500))
    X = madelon.loc[:, columns_x]
    y = madelon.loc[:, 500]
    X = np.array(X)
    y = np.array(y)
    en_one = OneHotEncoder()
    T = en_one.fit_transform(y.reshape(-1, 1)).toarray()

    return Dataset_split(X, T, 1)

# https://archive.ics.uci.edu/ml/datasets/MAGIC+Gamma+Telescope
def load_magic_dataset():
    magic = pd.read_table('../dataset/magic.csv', sep=',', header=None)
    #magic.to_csv('../dataset/magic.csv', header=None, index=False)
    columns_x = list(range(10))
    X = magic.loc[:, columns_x]
    y = magic.loc[:, 10]
    X = np.array(X)
    y = np.array(y)
    en_one = OneHotEncoder()
    T = en_one.fit_transform(y.reshape(-1, 1)).toarray()

    return Dataset_split(X, T, 1)

# https://archive.ics.uci.edu/ml/datasets/Mammographic+Mass
def load_mammo_dataset():
    mammo = pd.read_table('../dataset/mammographic_masses.csv', sep=',', header=None)
    columns_x = list(range(4))
    X = mammo.loc[:, columns_x]
    y = mammo.loc[:, 4]
    X = np.array(X)
    y = np.array(y)
    en_one = OneHotEncoder()
    T = en_one.fit_transform(y.reshape(-1, 1)).toarray()

    return Dataset_split(X, T, 1)

# https://archive.ics.uci.edu/ml/datasets/MONK%27s+Problems
def load_monk_dataset():
    monk = pd.read_table('../dataset/monks.csv', sep=',', header=None)
    #monk.to_csv('../dataset/monks_1.csv', header=None, index=False)
    columns_x = list(range(6))
    X = monk.loc[:, columns_x]
    y = monk.loc[:, 6]
    X = np.array(X)
    y = np.array(y)
    en_one = OneHotEncoder()
    T = en_one.fit_transform(y.reshape(-1, 1)).toarray()

    return Dataset_split(X, T, 1)

# https://archive.ics.uci.edu/ml/datasets/Musk+%28Version+2%29
def load_musk_dataset():
    musk = pd.read_table('../dataset/musk2.csv', sep=',', header=None)
    columns_x = list(range(166))
    X = musk.loc[:, columns_x]
    y = musk.loc[:, 166]
    X = np.array(X)
    y = np.array(y)
    en_one = OneHotEncoder()
    T = en_one.fit_transform(y.reshape(-1, 1)).toarray()

    return Dataset_split(X, T, 1)

# https://archive.ics.uci.edu/ml/datasets/Phishing+Websites
def load_phishing_dataset():
    puish = pd.read_table('../dataset/phishing.csv', sep=',', header=None)
    columns_x = list(range(30))
    X = puish.loc[:, columns_x]
    y = puish.loc[:, 30]
    X = np.array(X)
    y = np.array(y)
    en_one = OneHotEncoder()
    T = en_one.fit_transform(y.reshape(-1, 1)).toarray()

    return Dataset_split(X, T, 1)

# https://archive.ics.uci.edu/ml/datasets/Spambase
def load_spambase_dataset():
    spambase = pd.read_table('../dataset/spambase.data', sep=',', header=None)
    columns_x = list(range(55))
    X = spambase.loc[:, columns_x]
    y = spambase.loc[:, 57]
    X = np.array(X)
    y = np.array(y)
    en_one = OneHotEncoder()
    T = en_one.fit_transform(y.reshape(-1, 1)).toarray()

    return Dataset_split(X, T, 1)

# https://archive.ics.uci.edu/ml/datasets/Statlog+%28Australian+Credit+Approval%29
def load_statlog_dataset():
    statlog = pd.read_table('../dataset/statlog.csv', sep=',', header=None)
    #statlog.to_csv('../dataset/statlog.csv', header=None, index=False)
    columns_x = list(range(12))
    X = statlog.loc[:, columns_x]
    y = statlog.loc[:, 12]
    X = np.array(X)
    y = np.array(y)
    en_one = OneHotEncoder()
    T = en_one.fit_transform(y.reshape(-1, 1)).toarray()

    return Dataset_split(X, T, 1)

# https://archive.ics.uci.edu/ml/datasets/Thoracic+Surgery+Data
def load_thoraricsurgery_dataset():
    thoraricsurgery = pd.read_table('../dataset/thoraricsurgery.csv', sep=',', header=None)
    columns_x = list(range(16))
    X = thoraricsurgery.loc[:, columns_x]
    y = thoraricsurgery.loc[:, 16]
    X = np.array(X)
    y = np.array(y)
    en_one = OneHotEncoder()
    T = en_one.fit_transform(y.reshape(-1, 1)).toarray()

    return Dataset_split(X, T, 1)

# https://archive.ics.uci.edu/ml/datasets/Blood+Transfusion+Service+Center
def load_transfusion_dataset():
    transfusion = pd.read_table('../dataset/transfusion.csv', sep=',', header=None)
    columns_x = list(range(4))
    X = transfusion.loc[:, columns_x]
    y = transfusion.loc[:, 4]
    X = np.array(X)
    y = np.array(y)
    en_one = OneHotEncoder()
    T = en_one.fit_transform(y.reshape(-1, 1)).toarray()

    return Dataset_split(X, T, 1)

def Dataset_xor(N_train, N_test):
    # ----------train data----------
    # -----x-----
    x = np.random.normal(0, 1, (N_train, 2))

    # -----y-----
    y = np.logical_xor(x[:, 0] > 0, x[:, 1] > 0)
    y = np.where(y, 1, 0)

    # ----------test data----------
    x_test = np.random.normal(0, 1, (N_test, 2))
    y_test = np.logical_xor(x_test[:, 0] > 0, x_test[:, 1] > 0)
    y_test = np.where(y_test, 1, 0)

    x = normalize(Add_data(x))
    x_test = normalize(Add_data(x_test))

    en_one = OneHotEncoder()
    y = en_one.fit_transform(y.reshape(-1, 1)).toarray()
    y_test = en_one.fit_transform(y_test.reshape(-1, 1)).toarray()

    return x, x_test, y, y_test


# --------------------regression dataset--------------------
# y为x向量的和
def Dataset_sum(N_train, N_test, m_init):
    # ----------train data----------
    # -----x-----
    x = []
    for i in range(N_train):
        x_temp = [random.uniform(1, 100) for _ in range(m_init)]
        x_temp.append(1.0)
        x.append(x_temp)
    x = np.array(x)
    # print(x)

    # -----noise-----
    noise = []
    for i in range(N_train):
        noise.append(random.uniform(1, 10))  
    noise = np.array(noise)
    # print(noise)

    # -----y-----
    y = np.sum(x, 1) + noise

    #y = x[:, 0] * x[:, 1]
    # print(y)

    # ----------test data----------
    x_test = []
    y_test = []
    for i in range(N_test):
        x_temp = [random.uniform(1, 100) for _ in range(m_init)]
        x_temp.append(1.0)
        x_test.append(x_temp)
        y_test.append(sum(x_temp))
    x_test = np.array(x_test)

    x = normalize(Add_data(x))
    x_test = normalize(Add_data(x_test))
    y_norm = np.sqrt(norm(y)**2 + norm(y_test)**2)

    return x, x_test, y/y_norm, y_test/y_norm

# y为x向量的开根和
def Dataset_sqrt_sum(N_train, N_test, m_init):
    # ----------train data----------
    # -----x-----
    x = []
    for i in range(N_train):
        x_temp = [random.uniform(1, 100) for _ in range(m_init)]
        x_temp.append(1.0)
        x.append(x_temp)
    x = np.array(x)
    # print(x)

    # -----noise-----
    noise = []
    for i in range(N_train):
        noise.append(random.uniform(1, 5))
    noise = np.array(noise)
    # print(noise)

    # -----y-----
    y = np.sum(x**0.5, 1) + noise
    # print(y)

    # ----------test data----------
    x_test = []
    y_test = []
    for i in range(N_test):
        x_temp = [random.uniform(1, 100) for _ in range(m_init)]
        x_temp.append(1.0)
        x_test.append(x_temp)
        y_test.append(np.sum(np.array(x_temp)**0.5))
    x_test = np.array(x_test)

    x = normalize(Add_data(x))
    x_test = normalize(Add_data(x_test))
    y_norm = np.sqrt(norm(y)**2 + norm(y_test)**2)

    return x, x_test, y/y_norm, y_test/y_norm

# y为x向量的平方和
def Dataset_square_sum(N_train, N_test, m_init):
    # ----------train data----------
    # -----x-----
    x = []
    for i in range(N_train):
        x_temp = [random.uniform(1, 10) for _ in range(m_init)]
        x_temp.append(1.0)
        x.append(x_temp)
    x = np.array(x)
    # print(x)

    # -----noise-----
    noise = []
    for i in range(N_train):
        noise.append(random.uniform(1, 10)) 
    noise = np.array(noise)
    # print(noise)

    # -----y-----
    y = np.sum(x**2, 1) + noise
    # print(y)

    # ----------test data----------
    x_test = []
    y_test = []
    for i in range(N_test):
        x_temp = [random.uniform(1, 10) for _ in range(m_init)]
        x_temp.append(1.0)
        x_test.append(x_temp)
        y_test.append(np.sum(np.array(x_temp)**2))
    x_test = np.array(x_test)

    return x, x_test, y, y_test

def Dataset_url(url):
    raw_data = urllib.request.urlopen(url)
    dataset = np.loadtxt(raw_data, delimiter=',')

    return dataset

# https://archive.ics.uci.edu/ml/datasets/Airfoil+Self-Noise
def load_airfoil_dataset():
    airfoil = pd.read_table('../dataset/airfoil_self_noise.dat', sep='\t', header=None)
    columns_x = list(range(5))
    X = airfoil.loc[:, columns_x]
    y = airfoil.loc[:, 5]
    X = np.array(X)
    y = np.array(y)

    return Dataset_split(X, y)

# https://archive.ics.uci.edu/ml/datasets/Air+Quality
def load_airquality_dataset():
    airfoil = pd.read_table('../dataset/AirQualityUCI.csv', sep=',', header=None)
    columns_x = list(range(12))
    X = airfoil.loc[:, columns_x]
    y = airfoil.loc[:, 12]
    X = np.array(X)
    y = np.array(y)

    return Dataset_split(X, y)

# https://archive.ics.uci.edu/ml/datasets/Condition+Based+Maintenance+of+Naval+Propulsion+Plants
def load_cbm_dataset():
    clm = np.loadtxt('../dataset/CbmData.txt')
    X = clm[:, :16]
    y = clm[:, 16]

    return Dataset_split(X, y)

# https://archive.ics.uci.edu/ml/datasets/Communities+and+Crime
def load_communities_dataset(na_flag):  # na_flag=1:对nan用0进行替代，=2:去除nan所在的行
    comm = pd.read_table('../dataset/communities.data', sep=',', na_values='?', header=None)
    columns_x = list(range(3)) + list(range(4, 127))
    x_origin = comm.loc[:, columns_x]

    if na_flag == 1:
        X = x_origin.fillna(0)
    else:
        X = x_origin.dropna(axis=1)
    y = comm.loc[:, 127]
    X = np.array(X)
    y = np.array(y)

    return Dataset_split(X, y)

# https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength
def load_concrete_dataset(na_flag):
    comm = pd.read_table('../dataset/concrete.csv', sep=',', header=None)
    columns_x = list(range(8))
    X = comm.loc[:, columns_x]
    y = comm.loc[:, 8]
    X = np.array(X)
    y = np.array(y)

    return Dataset_split(X, y)

# https://archive.ics.uci.edu/ml/datasets/Communities+and+Crime+Unnormalized
def load_crime_dataset(na_flag, num):  # na_flag=1:对nan用0进行替代，=2:去除nan所在的行
    crime = pd.read_table('../dataset/CommViolPredUnnormalizedData.txt', sep=',', na_values='?', header=None)
    columns_x = list(range(2, num))
    x_origin = crime.loc[:, columns_x]

    if na_flag == 1:
        X = x_origin.fillna(0)
    else:
        X = x_origin.dropna(axis=1)
    y = crime.loc[:, num].fillna(0)
    X = np.array(X)
    y = np.array(y)

    return Dataset_split(X, y)

# https://archive.ics.uci.edu/ml/datasets/Daily+Demand+Forecasting+Orders
def load_demand_dataset():
    demand = pd.read_table('../dataset/Daily_Demand_Forecasting_Orders.csv', sep=';', header=None)
    columns_x = list(range(12))
    X = demand.loc[:, columns_x]
    y = demand.loc[:, 12]
    X = np.array(X)
    y = np.array(y)

    return Dataset_split(X, y)

# https://archive.ics.uci.edu/ml/datasets/Real+estate+valuation+data+set
def load_estate_dataset():
    estate = pd.read_table('../dataset/Real_estate_valuation_dataset.csv', sep=',', header=None)
    columns_x = list(range(5))
    X = estate.loc[:, columns_x]
    y = estate.loc[:, 5]
    X = np.array(X)
    y = np.array(y)

    return Dataset_split(X, y)

# https://archive.ics.uci.edu/ml/datasets/Facebook+metrics
def load_facebook_dataset():
    estate = pd.read_table('../dataset/facebook.csv', sep=',', header=None)
    columns_x = list(range(16))
    X = estate.loc[:, columns_x]
    y = estate.loc[:, 16]
    X = np.array(X)
    y = np.array(y)

    return Dataset_split(X, y)

# https://archive.ics.uci.edu/ml/datasets/QSAR+aquatic+toxicity
def load_qsar_dataset():
    estate = pd.read_table('../dataset/qsar_aquatic_toxicity.csv', sep=',', header=None)
    columns_x = list(range(8))
    X = estate.loc[:, columns_x]
    y = estate.loc[:, 8]
    X = np.array(X)
    y = np.array(y)

    return Dataset_split(X, y)

# https://archive.ics.uci.edu/ml/datasets/Bias+correction+of+numerical+prediction+model+temperature+forecast
def load_temperature_dataset():
    estate = pd.read_table('../dataset/temperature.csv', sep=',', header=None)
    columns_x = list(range(21))
    X = estate.loc[:, columns_x]
    y = estate.loc[:, 22]
    X = np.array(X)
    y = np.array(y)

    return Dataset_split(X, y)

# https://archive.ics.uci.edu/ml/datasets/Wine+Quality
def load_wine_dataset():
    wine = pd.read_table('../dataset/winequality-white.csv', sep=',', header=None)
    columns_x = list(range(11))
    X = wine.loc[:, columns_x]
    y = wine.loc[:, 11]
    X = np.array(X)
    y = np.array(y)

    return Dataset_split(X, y)


if __name__ == '__main__':
    print(load_test_dataset())