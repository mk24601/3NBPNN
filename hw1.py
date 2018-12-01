# 2入力1出力
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import *

B = 1.0 # シグモイド関数の中にあるやつ
N = 5 # 中間層のノード数
ETA = 0.5 # 学習率
EPOCH = 10000 # エポック数

# 重みとバイアスの初期化
def init_network(n):
    network = {}
    network['W1'] = np.random.random_sample((2, n))
    network['b1'] = np.random.random_sample((1, n))
    network['W2'] = np.random.random_sample((n, 1))
    network['b2'] = np.random.random_sample((1, 1))

    return network

# 前向き計算
def forward(network, x):
    W1, W2 = network['W1'], network['W2']
    b1, b2 = network['b1'], network['b2']
    # 1層目
    a1 = np.dot(x, W1) + b1 # A = XW + B
    y = sigmoid(a1) # Z = h(A)
    # 2層目
    a2 = np.dot(y, W2) + b2 # A = XW + B
    z = sigmoid(a2) # Z = h(A)

    return y[0], z[0]

# シグモイド関数(活性化関数)
def sigmoid(x):
    return 1 / (1 + np.exp(-1 * B * x))

# 誤差関数
def erf(t_data,z_data):
    J = 0
    for i in range(len(t_data)):
        J += (t_data[i] - z_data[i]) * (t_data[i] - z_data[i])
    J /= 2
    return J

# エラー率確認用関数
def validity(network):
    testdata = generate_data()
    t_data = []
    z_data = []
    for ky,val in testdata.items():
        t_data.append(val)
        z_data.append(forward(network,ky)[1])
    return(erf(t_data,z_data))

# 学習したい関数(今回はXOR)
def c_function(x, y):
    return x^y

# 教師データ作成
def generate_data():
    data = {}   
    data[(0,0)]= c_function(0,0)
    data[(0,1)]= c_function(0,1)
    data[(1,0)]= c_function(1,0)
    data[(1,1)]= c_function(1,1)
    return data

# 学習
def training(network,t_data,eta):
    t = 0
    y = [] 

    for e in range(EPOCH):
        for t in t_data.keys():
            y_data, z_data = forward(network, t)
            # update
            for j in range(len(network['W2'])):
                for k in range(len(network['W2'][j])):
                    network['W2'][j][k] += eta * (t_data[t] - z_data)*(z_data*(1 - z_data))*y_data[j]

            for j in range(len(network['b2'])):
                for k in range(len(network['b2'][j])):
                    network['b2'][j][k] += eta * (t_data[t] - z_data)*(z_data*(1 - z_data))

            for i in range(len(network['W1'])):
                for j in range(len(network['W1'][i])):   
                    tmp = network['W2'][j]*(t_data[t] - z_data)*(z_data*(1 - z_data))
                    network['W1'][i][j] += eta * tmp *y_data[j]*(1-y_data[j])*t[i]
            
            for i in range(len(network['b1'])):
                for j in range(len(network['b1'][i])):
                    tmp = network['W2'][j]*(1-y_data[j])*y_data[j]
                    network['b1'][i][j] += eta * tmp * (t_data[t] - z_data)*(z_data*(1 - z_data))

        # 誤差計算
        val = validity(network)
        y.append(val)
    return y

if __name__ == "__main__":
    
    y = []
    network = init_network(N)
    traindata = generate_data()
    y=training(network, traindata,ETA)

    # 最終的なモデルにXORの4パターン(0,0)(0,1)(1,0)(1,1)を入力してみた結果を表示
    last = generate_data()
    print("[0 0], [0 1], [1 0], [0 1]")
    for l in last.keys():
        print(str(forward(network,l)[1][0]),end="")
        if l == (1,1):
            print("")
        else :
            print(",",end="")
                

    # plot
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    x = [i for i in range(EPOCH)]
    plt.xlabel('epoch')
    plt.plot(x, y, label='error')

    plt.legend()
    plt.show()


    
