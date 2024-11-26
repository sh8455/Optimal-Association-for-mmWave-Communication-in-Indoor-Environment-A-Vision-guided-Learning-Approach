import numpy as np
import pandas as pd

df = pd.read_csv('DRL\predictions.csv')

BS_num = 4

x_column  = df['a']
x_list = x_column.tolist()
y_column = df['d']
y_list = y_column.tolist()
z_column = df['c']
z_list = z_column.tolist()

BS_Coordinates = [
    [19.5, -19.5, 2.8],
    [19.5, 19.5, 2.8],
    [-19.5, -19.5, 2.8],
    [-19.5, 19.5, 2.8]
]

K = [
    [1511.113, 0, 550],
    [0, 609.94, 222],
    [0, 0, 1]
]

Camera_extrinsic_matrix = [
    [
        [0.7132506, -0.0854193, -0.6956847, 19.5],
        [0, 0.9925461, -0.1218693, 2.8],
        [0.7009092, 0.08692335, 0.7079341, -19.5]
    ],
    [
        [-0.7071071, -0.08617459, -0.7018362, 19.5]
        [-3.72529E-09, 0.9925461, -0.1218693, 2.8],
        [0.7071069, -0.0861746, -0.7018365, 19.5],
    ],
    [
        [0.7132506, 0.0854193, 0.6956847, -19.5],
        [0, 0.9925461, -0.1218693, 2.8],
        [-0.7009092, 0.08692335, 0.7079341, -19.5]
    ],
    [
        [-0.7071071, 0.08617459, 0.7018362, -19.5],
        [3.72529E-09, 0.9925461, -0.1218693, 2.8],
        [-0.7071069, -0.0861746, -0.7018365, 19.5]
    ]
]

def cal_3DCoor(x,y,depth,K):
    Z = depth
    X = (x - K[0, 2]) * Z / K[0, 0]
    Y = (y - K[1, 2]) * Z / K[1, 1]
    return np.array([X, Y, Z])
    
def triangulate_3D_points(RT):
    A = []
    
    for i in range(BS_num):
        x,y = x_list[i], y_list[i]
        depth = z_list[i]
        
        P = cal_3DCoor(x,y,depth,K)
        
        R = RT[:, :3]
        T = RT[:, 3]
        
        A.append(R@P+T)
    A = np.array(A)
    U, S, Vt = np.linalg.svd(A)
    X = Vt[-1, :] / Vt[-1, -1]
    
    return X[:3]
    
    

