"""Algorithm - B-GCTD"""
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
import gurobipy as gp
from gurobipy import *
import math
import sys
import xlwt


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

    # # 绘制散点图
    # plt.scatter(Location_EV[:, 0], Location_EV[:, 1], c=labels)
    # plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='x')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.title('Clustering Result')

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
def Simulation(Q_input, cluster_indices, cluster_centers, Demand_point_num, Days, EV_num, q):
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
    charge_request_for_station = np.zeros((Q_input, Days))
    charge_request_for_station_filled = np.zeros((Q_input, Days))
    charge_request_for_station_filled_no_queue = np.zeros((Q_input, Days))
    for i in range(Q_input):
        tem = cluster_indices[i]
        for j in range(len(tem)):
            Total_Distance += total_charge_EV_num[tem[j]] * (abs(Location_EV[tem[j], 0] - cluster_centers[i, 0]) + abs(
                Location_EV[tem[j], 1] - cluster_centers[i, 1]))
            charge_request_for_station[i, :] += charge_EV_num[tem[j], :]
            charge_request_for_station_filled[i, :] = charge_request_for_station[i, :] - \
                                                          np.maximum(charge_request_for_station[i, :] - 2 * q[i + 1].X,
                                                                     np.zeros((1, Days)))
            charge_request_for_station_filled_no_queue[i, :] = charge_request_for_station[i, :] - \
                                                                   np.maximum(
                                                                       charge_request_for_station[i, :] - q[i + 1].X,
                                                                       np.zeros((1, Days)))


    Cost_transport = f_d * Total_Distance
    Cost_charge_power_fin = (Cost_charge_power + f_ch * Total_Distance) * (np.sum(charge_request_for_station_filled, axis=None)
                                                 / np.sum(charge_request_for_station, axis=None))
    DSR = np.sum(charge_request_for_station_filled_no_queue, axis=None) / np.sum(charge_request_for_station, axis=None)
    SR = np.sum(charge_request_for_station_filled, axis=None) / np.sum(charge_request_for_station, axis=None)

    return Cost_transport, Cost_charge_power_fin, DSR, SR
def GCTD_algorithm(Q_input):
    """求解一个给定 Q 的 LA 问题"""
    r_alpha_mean = Range_EV_quantile(0.5)
    n_alpha_mean = 10 * np.exp(- lamda ** 2 * (r_alpha_mean - 20) ** 2)
    r_alpha = Range_EV_quantile(1 - beta)
    n_alpha = 10 * np.exp(- lamda ** 2 * (r_alpha - 20) ** 2)
    num_clusters = Q_input
    max_points_per_cluster = math.floor(16 / n_alpha)  # 需要计算

    if max_points_per_cluster * Q_input < Demand_point_num:
        [cluster_indices, cluster_centers] = cluster(num_clusters, 10000)

    if max_points_per_cluster * Q_input >= Demand_point_num:
        [cluster_indices, cluster_centers] = cluster(num_clusters, max_points_per_cluster)

    S = list(range(1, Q_input + 1))
    A = {i: len(cluster_indices[i - 1]) for i in S}

    # Modeling
    mdl = Model('P1')

    # 设置参数，禁用输出信息
    mdl.setParam("OutputFlag", 0)

    # Decision variables
    q = mdl.addVars(S, vtype=GRB.INTEGER)
    z = mdl.addVars(S, vtype=GRB.CONTINUOUS)

    # Objective function
    mdl.modelSense = GRB.MINIMIZE
    mdl.setObjective(quicksum(z[i] for i in S))

    # Constraints
    # mdl.addConstrs(2 * q[i] >= A[i] * n_alpha for i in S)
    mdl.addConstrs(q[i] >= 1 for i in S)
    mdl.addConstrs(q[i] <= 8 for i in S)
    mdl.addConstrs(math.ceil(A[i] * n_alpha) - 2 * q[i] <= z[i] for i in S)
    mdl.addConstrs(2 * q[i] - math.ceil(A[i] * n_alpha) <= z[i] for i in S)
    mdl.addConstrs(z[i] >= 0 for i in S)
    if max_points_per_cluster * Q_input < Demand_point_num:
        mdl.addConstrs(q[i] == 8 for i in S)

    mdl.optimize()

    Total_q = sum(q[i].X for i in S)

    # 计算总距离
    Total_Distance = 0
    for i in range(Q_input):
        tem = cluster_indices[i]
        for j in range(len(tem)):
            Total_Distance += abs(Location_EV[tem[j], 0] - cluster_centers[i, 0]) + abs(Location_EV[tem[j], 1]
                                                                                            - cluster_centers[i, 1])

    [Cost_transport, Cost_charge_power, DSR, SR] = Simulation(Q_input, cluster_indices, cluster_centers, Demand_point_num, Days, EV_num, q)  # 系统仿真

    Cost_1 = (f_c * Q_input + f_m * Total_q) * Years
    Cost_2 = (f_d * Total_Distance * n_alpha_mean) * Days + f_ch * Total_Distance * Days
    Cost_3 = (f_ch * (250 - r_alpha_mean) * 1079 * n_alpha_mean) * Days

    Total_cost = (Cost_1 + Cost_2 + Cost_3 * (SR + (1 - SR) * penalty)) / 1000

    return Total_cost, SR, DSR

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
f_c = 5000
f_m = 500
f_d = 0.041
f_ch = 0.0388
penalty = 2
np.random.seed(1)
beta = 0.6

"""Bisection search"""
delta = 5
Q_upper = Demand_point_num
Q_lower = 1
iteration = 1

while 1:

    Q_i = math.floor((Q_upper + Q_lower) / 2)

    if Q_i - Q_lower - delta > 0:
        a = Q_i - delta
    else:
        a = Q_i - 1

    if Q_upper - Q_i - delta > 0:
        b = Q_i + delta
    else:
        b = Q_i + 1

    [Cost_minus, DR_minus, SDR_minus] = GCTD_algorithm(a)
    [Cost_plus, DR_plus, SDR_plus] = GCTD_algorithm(b)

    if Cost_minus > Cost_plus:
        Q_lower = Q_i
    if Cost_minus <= Cost_plus:
        Q_upper = Q_i

    if Q_upper - Q_lower <= 1:
        Q_optimal = Q_upper
        Cost_optimal = Cost_plus
        DR_output = DR_plus
        SDR_output = SDR_plus
        break

    print("Iteration:", iteration)
    print("Lower bound:", Q_lower)
    print("Upper bound:", Q_upper)
    iteration += 1

print(Q_optimal)
