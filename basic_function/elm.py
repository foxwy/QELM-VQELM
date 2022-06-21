#/
#model
#/
import numpy as np
import math
import scipy

'''
@function:StoreWeights
@description:将参数保存在电脑里
@input:data:data:待保存的数据,类型随意，FileName:保存成的文件名，诸如'XX.txt'
'''
def StoreWeights(data, FileName):
    import pickle
    fw = open(FileName, 'wb')
    pickle.dump(data, fw)
    fw.close()

'''
@function:GrabWeights
@description:将电脑保存的数据提取
@input:FileName:已保存的数据文件名，诸如'XX.txt'
@output:提取的保存文件
'''
def GrabWeights(FileName):
    import pickle
    fr = open(FileName, 'rb')
    return pickle.load(fr)

#/
#随机选择数据来作为训练集，测试集和预测集，移除部分数据
#/
def random_choose(train_num, test_num, predict_num, remove_nums):
    # 提取训练测试编号
    dataSet = list(range(train_num + test_num + predict_num + len(remove_nums)))
    for i in remove_nums:
        dataSet.remove(i)
    trainSet = []; testSet = []; predictSet = []  # 随机提取的数据

    for i in range(train_num):
        randIndex = int(np.random.uniform(0, len(dataSet)))
        trainSet.append(dataSet[randIndex])
        del(dataSet[randIndex])

    for i in range(test_num):
        randIndex = int(np.random.uniform(0, len(dataSet)))
        testSet.append(dataSet[randIndex])
        del(dataSet[randIndex])

    for i in range(predict_num):
        randIndex = int(np.random.uniform(0, len(dataSet)))
        predictSet.append(dataSet[randIndex])
        del(dataSet[randIndex])
    return trainSet, testSet, predictSet

#/
#归一化
#/
def normalization(data, max, min):
    if len(data) != 0:
        m, n = len(data), len(data[0])
        print(m, n)
        for i in range(m):
            for j in range(n):
                data[i][j] = 0.2 + 0.6 * (data[i][j] - min) / (max - min)
    return data

#/
#sigmoid函数
#/
def sigmoid(a, x):
    '''
    定义Sigmoid函数: g(z) = 1 / (1 + e^-(ax + b))
    '''
    return 1.0 / (1 + np.exp(-1.0 * (x.dot(a))))

#/
#极限学习机的训练
#/
def ELM_train(X, T, C, L):
    # 随机初始化
    a = np.random.normal(-1, 1, (len(X[0]), L))

    # 使用特征映射求解输出矩阵
    H = sigmoid(a, X)

    # 计算输出权重和输出函数
    if L <= len(X):
        beta = scipy.linalg.pinv(H.T.dot(H)+np.identity(L)/C).dot(H.T.dot(T))
    else:
        beta = H.T.dot(scipy.linalg.pinv(H.dot(H.T)+np.identity(len(X))/C).dot(T))

    # 返回计算结果
    return H.dot(beta), a, beta

#/
#极限学习机的预测
#/
def ELM_predict(X, a, beta):
    H = sigmoid(a, X)
    return H.dot(beta)