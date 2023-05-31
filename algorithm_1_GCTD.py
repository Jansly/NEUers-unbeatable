from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
from gurobipy import *
import math
import sys

# 读取EV位置信息
path = r'D:\Research\【Cm】2023.02-MOPTA Modeling Competition\MOPTA_2023_data_demo\MOPTA2023_car_locations_data.xlsx'
dataframe = pd.read_excel(path, sheet_name='Location')
Location_EV = dataframe.values

def cluster(num_clusters, max_points_per_cluster):
    """给定聚类个数及每类最大容量，对需求点进行分类"""
    kmeans = KMeans(n_clusters=num_clusters, max_iter=300, random_state=0)
    kmeans.fit(Location_EV)

    # 获取每个点所属的类别
    labels = kmeans.labels_

    # 获取每个类别中的点的索引
    cluster_indices = [[] for _ in range(num_clusters)]
    for i, label in enumerate(labels):
        if len(cluster_indices[label]) < max_points_per_cluster:
            cluster_indices[label].append(i)

    # 获取每类的中心坐标
    cluster_centers = kmeans.cluster_centers_

    # 绘制散点图
    plt.scatter(Location_EV[:, 0], Location_EV[:, 1], c=labels)
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='x')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Clustering Result')
    # plt.show()
    plt.savefig('output.png', dpi=300)

    return cluster_indices, cluster_centers
def Range_EV_quantile(alpha):
    """计算截断正态分布分位数"""
    lower_bound = 20  # 下限
    upper_bound = 250  # 上限
    mean = 100  # 均值
    std = 50  # 标准差

    # 创建截断正态分布对象
    dist = truncnorm((lower_bound - mean) / std, (upper_bound - mean) / std, loc=mean, scale=std)

    # 计算分位数
    quantile = dist.ppf(alpha)

    return quantile
def Simulation(Demand_point_num, Days, EV_num, q):
    """仿真运行"""
    charge_EV_num = np.zeros((Demand_point_num, Days))
    Cost_charge_power = 0
    for t in range(EV_num):
        size = Demand_point_num * Days
        lower, upper = (lower_bound - mean) / std, (upper_bound - mean) / std
        samples = truncnorm.rvs(lower, upper, loc=mean, scale=std, size=size)
        A = np.reshape(samples, (Demand_point_num, Days))
        B = np.exp(- lamda**2 * (A - 20)**2)    # 充电概率
        C = np.random.rand(Demand_point_num, Days)      # 生成随机数

        comparison = np.less_equal(C, B)  # 逐元素比较，返回布尔值数组（是否充电）
        result = comparison.astype(int)  # 将布尔值数组转换为整数数组

        charge_power_request = np.multiply(A, result)
        total_charge_power_request = np.sum(charge_power_request, axis=None)
        Cost_charge_power += f_ch * (250 * np.sum(result, axis=None) - total_charge_power_request)

        charge_EV_num += result
    total_charge_EV_num = np.sum(charge_EV_num, axis=1)

    Total_Distance = 0
    charge_request_for_station = np.zeros((Q, Days))
    charge_request_for_station_filled = np.zeros((Q, Days))
    charge_request_for_station_filled_no_queue = np.zeros((Q, Days))
    for i in range(Q):
        tem = cluster_indices[i]
        for j in range(len(tem)):
            Total_Distance += total_charge_EV_num[tem[j]] * (abs(Location_EV[tem[j], 0] - cluster_centers[i, 0]) + abs(
                Location_EV[tem[j], 1] - cluster_centers[i, 1]))
            charge_request_for_station[i, :] += charge_EV_num[tem[j], :]
        charge_request_for_station_filled[i, :] = charge_request_for_station[i, :] - \
                                                  np.maximum(charge_request_for_station[i, :] - 2 * q[i+1].X, np.zeros((1, Days)))
        charge_request_for_station_filled_no_queue[i, :] = charge_request_for_station[i, :] - \
                                                  np.maximum(charge_request_for_station[i, :] - q[i + 1].X,
                                                             np.zeros((1, Days)))

    Cost_transport = f_d * Total_Distance
    Cost_charge_power_fin = (Cost_charge_power + f_ch * Total_Distance) * (np.sum(charge_request_for_station_filled, axis=None)
                                                 / np.sum(charge_request_for_station, axis=None))
    DSR = np.sum(charge_request_for_station_filled_no_queue, axis=None) / np.sum(charge_request_for_station, axis=None)
    SR = np.sum(charge_request_for_station_filled, axis=None) / np.sum(charge_request_for_station, axis=None)

    return Cost_transport, Cost_charge_power_fin, DSR, SR

# 问题参数
Demand_point_num = 1079
EV_num = 10
mean = 100
std = 50
upper_bound = 250
lower_bound = 20
Years = 5
Days = Years * 365
lamda = 0.012
Q = 50
f_c = 5000
f_m = 500
f_d = 0.041
f_ch = 0.0388

beta = 0.1

np.random.seed(1)

r_alpha_mean = Range_EV_quantile(0.5)
n_alpha_mean = 10 * np.exp(- lamda**2 * (r_alpha_mean - 20)**2)

r_alpha = Range_EV_quantile(1-beta)
n_alpha = 10 * np.exp(- lamda**2 * (r_alpha - 20)**2)

num_clusters = Q
max_points_per_cluster = math.floor(16 / n_alpha)     # 需要计算

if max_points_per_cluster * Q < Demand_point_num:
    print("Infeasible setting, maximum covered demand point number:", max_points_per_cluster * Q)
    sys.exit()

[cluster_indices, cluster_centers] = cluster(num_clusters, max_points_per_cluster)

S = list(range(1, Q + 1))
A = {i: len(cluster_indices[i-1]) for i in S}

# Modeling
mdl = Model('P1')

# Decision variables
q = mdl.addVars(S, vtype=GRB.INTEGER)    # Selection of warehouse

# Objective function
mdl.modelSense = GRB.MINIMIZE
mdl.setObjective(quicksum(q[i] for i in S))

# Constraints
mdl.addConstrs(q[i] >= A[i] * n_alpha / 2 for i in S)
mdl.addConstrs(q[i] >= 1 for i in S)
mdl.addConstrs(q[i] <= 8 for i in S)

mdl.optimize()

Total_q = sum(q[i].X for i in S)

# 计算总距离
Total_Distance = 0
for i in range(Q):
    tem = cluster_indices[i]
    for j in range(len(tem)):
        Total_Distance += abs(Location_EV[tem[j], 0] - cluster_centers[i, 0]) + abs(Location_EV[tem[j], 1]
                                                                                    - cluster_centers[i, 1])

Cost_1 = (f_c * Q + f_m * Total_q) * Years
Cost_2 = (f_d * Total_Distance * n_alpha_mean) * Days
Cost_3 = (f_ch * ((250 - r_alpha_mean) * 1079 * n_alpha_mean + Total_Distance)) * Days

Total_Cost = Cost_1 + Cost_2 + Cost_3

print("Time average Total Cost:", (Total_Cost / 1000))
print("Transportation Cost:", (Cost_2 / 1000))
print("Charging Cost:", (Cost_3 / 1000))

print("Total number of chargers:", Total_q)
# print(Total_Distance)

# # 系统仿真 - 5 years
# [Cost_transport, Cost_charge_power, DSR, SR] = Simulation(Demand_point_num, Days, EV_num, q)     # 系统仿真
#
# print("Total Cost:", ((Cost_1 + Cost_transport + Cost_charge_power) / 1000))
# print("Construction Cost:", (Cost_1 / 1000))
# print("Simulation - Transport Cost:", (Cost_transport / 1000))
# print("Simulation - Charging Cost:", (Cost_charge_power / 1000))
# print("DSR:", DSR * 100)
# print("SR:", SR * 100)


