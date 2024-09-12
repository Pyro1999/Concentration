# # import matplotlib.pyplot as plt
# # from mpl_toolkits.mplot3d import Axes3D
# import numpy as np
# # import matplotlib
# from sklearn import datasets
# # #
# # # # def add(a: int) -> int:
# # # #     return a + 1
# # #
# # # # a=sorted([f.name for f in matplotlib.font_manager.fontManager.ttflist])
# # # #
# # # # for i in a:
# # # #     print(i)
# # #
# # #
# # # # 创建数据
# # # x = np.linspace(-5, 5, 100)
# # # y = np.linspace(-5, 5, 100)
# # # x, y = np.meshgrid(x, y)
# # # z = np.sin(np.sqrt(x**2 + y**2))
# # #
# # # # n = add(1.5)
# # # # print(n)
# # #
# # # a = [1,3,5]
# # # b = [0,2]
# # # a[b] += 1
# # # print(a)
# # # # 创建图形
# # # fig = plt.figure()
# # # ax = fig.add_subplot(111, projection='3d')  # 添加带有三维投影的坐标轴
# # #
# # # # 绘制三维曲面图
# # # ax.plot_surface(x, y, z, cmap='viridis')
# # #
# # # # 设置坐标轴标签
# # # ax.set_xlabel('X Label')
# # # ax.set_ylabel('Y Label')
# # # ax.set_zlabel('Z Label')
# # #
# # # # 显示图形
# # # plt.show()
# #
#
# # import numpy as np
# #
# # class Particle:
# #     def __init__(self, dim):
# #         self.position = np.random.rand(dim)  # 初始化粒子位置
# #         self.velocity = np.random.rand(dim)  # 初始化粒子速度
# #         self.best_position = self.position.copy()  # 记录粒子历史最优位置
# #         self.best_score = float('inf')  # 记录粒子历史最优适应度
# #
# # class PSO:
# #     def __init__(self, num_particles, dim, max_iter, objective_func):
# #         self.num_particles = num_particles
# #         self.dim = dim
# #         self.max_iter = max_iter
# #         self.objective_func = objective_func
# #         self.particles = [Particle(dim) for _ in range(num_particles)]  # 初始化粒子群
# #
# #     def optimize(self):
# #         global_best_position = None
# #         global_best_score = float('inf')
# #
# #         for _ in range(self.max_iter):
# #             for particle in self.particles:
# #                 score = self.objective_func(particle.position)
# #                 if score < particle.best_score:
# #                     particle.best_score = score
# #                     particle.best_position = particle.position.copy()
# #                 if score < global_best_score:
# #                     global_best_score = score
# #                     global_best_position = particle.position.copy()
# #
# #             for particle in self.particles:
# #                 inertia_weight = 0.5  # 惯性权重
# #                 cognitive_weight = 1.5  # 认知权重
# #                 social_weight = 1.5  # 社会权重
# #
# #                 cognitive_component = cognitive_weight * np.random.rand(self.dim) * (particle.best_position - particle.position)
# #                 social_component = social_weight * np.random.rand(self.dim) * (global_best_position - particle.position)
# #                 particle.velocity = inertia_weight * particle.velocity + cognitive_component + social_component
# #                 particle.position += particle.velocity
# #
# #         return global_best_position, global_best_score
# #
# # # 示例：优化目标函数 f(x) = x^2
# # def objective_function(x):
# #     return np.sum(x ** 2)
# #
# # # 设置参数
# # num_particles = 20
# # dim = 5
# # max_iter = 100
# # pso = PSO(num_particles, dim, max_iter, objective_function)
# #
# # # 执行优化
# # best_position, best_score = pso.optimize()
# # print("Best Position:", best_position)
# # print("Best Score:", best_score)
#
# # x = np.array([[2,3], [4,5], [6,7]])
# # print(x.ravel())
# # iris = datasets.load_iris()
# # x = iris['data']
# # print('1', iris['data'])
# # y = iris['target']
# # sorv = (y==0)|(y==1)
# # # print(y)
# # # print(sorv)
# # print('2', x[sorv])
# #
# # x = np.linspace(0, 5, 5)
# # y = np.linspace(6, 10, 5)
# # x, y = np.meshgrid(x, y)
# # print(x.ravel(), y.ravel())
# # X = np.c_[x.ravel(), y.ravel()]
# # print(X)
# # a = np.array([23,4,5])
# # print(range(1,10))
# #
# # b = None
# # b = [0,2,3]
# #
# # print(b)
#
# # x = np.array([23.5, 64])
# # print(x.shape)
# # print(np.array([[12,33],[213,34],[33,131]]).shape[1])
#

import os
import time
import matplotlib.pyplot as plt
import xml.dom.minidom
import numpy as np
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import sklearn
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from pybaselines import Baseline
from math import sqrt
import random
import matplotlib as mpl
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

from LSSVM import LSSVMRegression


# 读取检测结果xml文件并取对数处理，方便观察研究
def read_xml(address):
    dom = xml.dom.minidom.parse(address)  # 读取xml文件
    root = dom.documentElement  # 获取xml文档对象，注意这里的root，他的对象类型是Element对象，并且是根节点。

    bond_list = root.getElementsByTagName('row')  # 获取根节点下所有标签名为row的节点，返回的是一个列表，列表内容也是Element对象
    str_count = bond_list[0].firstChild.data.split(",")  # 将bond_list里row元素提取出来并通过split组装成一个list，即为
    count = [int(x) for x in str_count]

    name_list = root.getElementsByTagName('Name')
    name = name_list[0].firstChild.data

    group_node = root.getElementsByTagName('group')[0]
    number_list = group_node.getElementsByTagName('index')[0]
    number = number_list.firstChild.data

    original_data = []  # 将Counts转换为Intensity(cps)(除以时间（s），这里是200s，可根据实际情况修改参数)
    for item in count:
        original_data.append(item / 200)
    original_data = original_data[299:1400]
    return original_data, name, number


# 把给定文件夹下的数据读取出来装载到每个元素的字典里并浓度大小升序排列
def load_data(address):
    dictionary = {}
    temp_data = []
    average_data = []
    files = os.listdir(address)
    temp_name = []
    names = []
    # files.sort(key=lambda x: float(x[3:-8]))  # 使读取的数据按浓度大小升序排列并只取自己输入的“Name”部分，去掉前方的“ScanXXXX...”
    for file in files:
        add = os.path.join(address, file)
        data1, name, number = read_xml(add)
        temp_data.append(data1)
        temp_name.append(name)

    for i in range(0, len(temp_data), 3):
        triplet = temp_data[i:i + 3]
        average = [0] * len(temp_data[0])  # Create a new list for each iteration
        for j in range(0, len(temp_data[0])):
            average[j] = float('%.4f' % ((triplet[0][j] + triplet[1][j] + triplet[2][j]) / 3))
        average_data.append(average)
        names.append(temp_name[i])

    for j in range(0, len(names)):
        dictionary[names[j]] = average_data[j]

    return dictionary


def load_cali_data(address):
    dictionary = {}
    files = os.listdir(address)
    # files.sort(key=lambda x: int(x[:-10]))  # 使读取的数据按浓度大小升序排列并只取自己输入的“Name”部分，去掉前方的“ScanXXXX...”
    for file in files:
        # print('file',file)
        add = os.path.join(address, file)
        # print('add',add)
        data, name, _ = read_xml(add)
        # print('data,name',data,name)
        dictionary[name] = data
        # print('dictionary',dictionary)
    return dictionary


# SG滤波函数，w是窗口长度，p是对窗口内数据进行的拟合阶数
def SG(d, w=7, p=3):
    return savgol_filter(d, w, p)


# 消除负峰
def refine(background, original_spectrum):
    b_1 = np.arange(len(original_spectrum), dtype=np.float64)
    for n in range(0, len(original_spectrum)):
        if background[n] < original_spectrum[n]:
            b_1[n] = original_spectrum[n]
        else:
            b_1[n] = background[n]
    return b_1


# 二阶导数寻峰
def D2(d):
    d = np.array([d])
    n, p = d.shape
    Di = np.ones((n, p - 2))
    for i in range(n):
        Di[i] = np.diff(np.diff(d[i]))
    return Di.T


# 计算峰面积(测试：使用峰值附近五道的count和作为面积)
def calc_area(pk):
    pass


# 计算calibration curve并且衡量性能(RMSE＆R2)
def calc_curve(a, c):
    lr = LinearRegression()
    x = a.reshape(-1, 1)
    y = c.reshape(-1, 1)
    lr.fit(x, y)
    w = lr.coef_[0][0]
    b = lr.intercept_[0]
    y_pred = lr.predict(x)
    # # print('concentration,x', concentration, x)
    return y_pred, w, b


# calibration curve性能度量
def get_rmse_r2(y, y_pred):
    total_cost = 0
    m = len(y)
    # 计算RMSE
    for k in range(m):
        total_cost += (y[k] - y_pred[k]) ** 2
    total_cost = sqrt(total_cost / m)
    # 计算R方
    r_2 = r2_score(y, y_pred)
    return total_cost, r_2


# 模拟退火算法
def simulated_annealing(counts, concentrations, w, b):
    # 初始化
    T = 100  # 初始温度
    MAX_EPOCH = 100  # 迭代次数
    LAMBDA = 0.9  # 退火速率
    END_TEMP = 0.1  # 结束温度
    CHANGE_NEIGHBORHOOD = 1  # 改变的邻域
    q = 0
    # 存储结果
    result = {}

    while T > END_TEMP:  # 未到达目标时
        new_x = []
        y = []
        x = 0
        # t = random.randint(130, 160)
        # 每个浓度随机一个解
        for count in counts:
            # print('i',i[random.randint(0,1)])
            if q != 0:
                x = sum(count[Cd_Ka - q:Cd_Ka + q + 1])
                new_x.append(x)
            else:
                x = count[Cd_Ka]
                new_x.append(x)
        # print('new_x,', new_x)
        for i in range(len(new_x)):
            y.append(w * new_x[i] + b)
        _, r = get_rmse_r2(np.array(concentrations), np.array(y))
        for epoch in range(MAX_EPOCH):
            # 生成新的解
            new_x = []
            new_y = []
            while True:
                new_q = q + random.randint(-CHANGE_NEIGHBORHOOD, CHANGE_NEIGHBORHOOD)
                if 0 <= new_q <= 30:
                    break  # 重复得到新解，直到产生的新解满足约束条件
            for count in counts:
                # print('i',i[random.randint(0,1)])
                if new_q != 0:
                    x = sum(count[Cd_Ka - new_q:Cd_Ka + new_q + 1])
                    new_x = np.append(new_x, x)
                else:
                    x = count[Cd_Ka]
                    new_x = np.append(new_x, x)
            for i in range(len(new_x)):
                new_y.append(w * new_x[i] + b)
            # print('new_y, y', new_y, y)
            _, new_r = get_rmse_r2(np.array(concentrations), np.array(new_y))
            if Metropolis(new_r - r, T):
                q = new_q
                r = new_r
        # 记录当前温度的结果
        # print('x,y', x, y)
        #        x = np.ndarray.tolist(x)
        # print('x', tuple(x))
        result[q] = r
        # 降温
        T *= LAMBDA
    # print('result', result.items())
    result_x, result_y = sorted(result.items(), key=lambda x: x[1], reverse=True)[0]
    print("q:", result_x, "此时得到的最佳通道选择范围为", (Cd_Ka - result_x + 300, Cd_Ka + result_x + 300),
          ",此时R方最大，为", result_y, ")")
    return result_x


# metropolis准则
def Metropolis(delta_f, T):
    if delta_f >= 0:
        return True
    else:
        return True if np.exp(-(delta_f / T)) >= random.uniform(0, 1) else False


def average_relative_error(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 防止除以零
    non_zero_mask = y_true != 0
    relative_errors = np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])

    are = np.mean(relative_errors)
    return are


def maximum_relative_error(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 防止除以零
    non_zero_mask = y_true != 0
    relative_errors = np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])

    mre = np.max(relative_errors)
    return mre


start_time = time.time()
# 0.设置光谱横轴——通道/能量
Channel = range(1, 2049)[299:1400]  # 仪器的2048个通道中第300道到第1400道（包含感兴趣的部分——Zn和Cd的有效峰）
Energy = [0.0197787 * x - 0.0483658 for x in Channel]  # 能量，光谱的横轴
Channel2 = np.arange(1, 2047, 1)[300:1399]
Energy2 = [0.0197787 * x - 0.0483658 for x in Channel2]  # 能量，光谱的横轴for导数(比Energy要少左右边界两个值)

# 1.读取实验原始数据
cali_data_1 = load_cali_data(r'D:\PycharmProjects\ion_concentration_detection_EDXRF\Cd_xml_data')
# print('cali1', cali_data_1)
Cd_1 = load_data(r'D:\PycharmProjects\ion_concentration_detection_EDXRF\Zn_Cd_xml_data')  # 读取测量数据
# print('Cd1', Cd_1)
Cd_purafication_1 = \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8178, 9378, 9466, 9806, 9627, 9598, 9530, 9831, 9585, 9434, 9511, 9363, 9480, 9185, 9156, 9053, 9148, 9041, 8930, 8870, 8681, 9053, 8587, 8846, 8632, 8783, 8628, 8831, 8729, 8807, 8613, 8574, 8607, 8605, 8686, 8497, 8350, 8373, 8207, 8216, 8241, 8197, 8052, 8427, 8406, 8511, 8457, 8413, 8506, 8185, 8051, 7870, 7882, 7814, 7709, 7797, 7623, 7923, 7768, 8011, 7884, 8074, 8390, 8637, 8948, 8893, 9078, 9043, 8992, 8999, 8893, 9128, 8875, 9069, 8938, 8807, 8975, 8771, 8942, 8840, 9037, 9378, 9532, 9853, 10645, 11882, 12564, 13339, 12677, 12128, 10839, 10195, 9521, 9400, 9311, 9445, 9473, 9378, 9243, 9308, 9198, 9054, 9209, 9258, 9208, 9184, 9221, 9153, 9014, 9184, 9016, 9165, 9353, 9184, 9231, 9310, 9220, 9270, 9218, 9437, 9313, 9538, 9335, 9470, 9399, 9281, 9135, 9362, 9290, 9317, 9239, 9290, 9371, 9301, 9624, 9712, 9775, 9954, 9889, 9793, 9660, 9299, 8881, 9008, 8731, 8620, 8734, 8513, 8362, 8407, 8579, 8365, 8256, 8409, 8252, 8116, 8050, 8130, 7777, 7823, 7563, 7292, 7203, 6940, 7063, 6993, 7024, 6801, 6851, 6593, 6429, 6365, 6155, 5686, 5574, 5328, 5325, 5257, 5201, 5125, 4989, 4862, 4806, 4522, 4410, 4411, 4278, 4294, 4183, 4076, 3877, 3790, 3771, 3647, 3580, 3373, 3331, 3288, 3272, 3202, 3071, 2830, 2907, 2822, 2756, 2721, 2574, 2624, 2451, 2512, 2321, 2329, 2271, 2184, 2122, 2070, 1985, 1965, 1992, 1842, 1853, 1812, 1735, 1663, 1598, 1574, 1628, 1492, 1447, 1404, 1443, 1376, 1399, 1267, 1164, 1182, 1183, 1155, 1074, 1043, 1018, 1056, 975, 1037, 975, 962, 939, 963, 895, 820, 885, 821, 834, 789, 861, 817, 811, 834, 851, 882, 1047, 1406, 2025, 3143, 4810, 7049, 9491, 11701, 12898, 12445, 11311, 9024, 6486, 4564, 2903, 1799, 1106, 942, 809, 725, 686, 691, 729, 713, 647, 681, 738, 758, 787, 936, 1096, 1313, 1582, 1855, 2217, 2594, 2935, 3140, 3247, 3092, 2722, 2240, 1948, 1437, 1115, 953, 805, 754, 731, 836, 888, 1080, 1314, 1817, 2230, 2835, 3541, 3985, 4311, 4209, 3932, 3268, 2686, 2008, 1533, 1275, 1079, 956, 899, 893, 860, 820, 816, 870, 855, 885, 808, 785, 864, 927, 885, 913, 983, 1119, 1274, 1477, 1694, 1709, 1764, 1694, 1636, 1515, 1207, 1074, 913, 848, 812, 780, 724, 794, 839, 937, 1069, 1120, 1247, 1263, 1319, 1311, 1348, 1345, 1293, 1296, 1270, 1359, 1479, 1563, 1856, 2081, 2492, 2816, 3230, 3375, 3669, 3880, 4044, 4096, 4288, 4506, 5003, 5351, 5981, 6937, 8112, 9850, 13099, 19137, 31929, 57329, 106288, 189395, 328977, 525541, 794605, 1096221, 1408285, 1682581, 1829546, 1860765, 1725908, 1497602, 1184158, 877287, 592033, 374725, 221950, 121290, 65236, 33278, 18064, 10791, 7770, 6446, 5739, 5396, 5127, 4942, 4695, 4527, 4451, 4547, 4534, 4556, 4802, 5101, 5456, 5755, 6304, 6883, 7688, 8523, 9213, 10737, 11945, 14425, 18552, 26177, 38672, 58130, 87584, 125540, 174143, 221650, 269154, 297878, 311391, 304581, 272165, 229974, 178075, 130999, 89138, 57381, 34477, 19783, 11142, 6200, 3624, 2315, 1627, 1317, 1171, 1069, 1089, 1013, 949, 958, 924, 853, 860, 798, 816, 766, 707, 750, 776, 693, 638, 706, 639, 687, 621, 633, 610, 642, 584, 632, 599, 570, 579, 589, 598, 557, 603, 604, 593, 608, 551, 539, 580, 557, 584, 530, 543, 487, 494, 538, 552, 593, 640, 618, 663, 671, 666, 678, 631, 587, 580, 580, 482, 461, 476, 480, 503, 498, 493, 507, 534, 498, 485, 523, 473, 502, 527, 475, 469, 494, 463, 456, 407, 415, 416, 400, 410, 362, 386, 392, 377, 416, 359, 387, 398, 423, 416, 425, 374, 389, 389, 368, 364, 371, 364, 384, 388, 378, 389, 381, 353, 352, 340, 358, 302, 349, 316, 322, 337, 317, 317, 323, 330, 300, 343, 317, 346, 334, 320, 316, 304, 309, 320, 331, 305, 308, 315, 300, 310, 312, 304, 292, 327, 315, 293, 330, 306, 304, 302, 321, 309, 306, 313, 288, 305, 277, 291, 315, 334, 279, 292, 273, 276, 289, 262, 293, 297, 318, 366, 473, 595, 797, 1100, 1387, 2002, 2742, 3341, 4249, 4820, 5426, 5664, 5630, 5481, 4940, 4276, 3599, 2894, 2119, 1608, 1134, 887, 679, 571, 448, 377, 366, 304, 293, 289, 259, 240, 266, 230, 226, 223, 253, 251, 253, 305, 344, 448, 504, 602, 830, 914, 1104, 1343, 1479, 1576, 1640, 1619, 1539, 1422, 1265, 1116, 961, 731, 547, 420, 342, 319, 243, 239, 244, 281, 265, 283, 264, 291, 302, 307, 301, 263, 279, 254, 257, 263, 271, 264, 306, 312, 385, 474, 595, 715, 900, 1097, 1199, 1264, 1385, 1422, 1468, 1284, 1220, 1038, 864, 765, 621, 498, 460, 379, 357, 359, 286, 343, 287, 267, 272, 247, 225, 232, 210, 235, 188, 195, 212, 216, 225, 190, 242, 231, 216, 231, 220, 242, 271, 286, 296, 335, 353, 378, 429, 422, 440, 447, 469, 418, 372, 365, 323, 283, 253, 266, 266, 209, 246, 225, 201, 217, 215, 231, 194, 172, 206, 195, 186, 179, 161, 160, 159, 156, 182, 196, 212, 177, 172, 192, 168, 217, 169, 189, 193, 205, 202, 201, 223, 209, 239, 233, 225, 241, 240, 241, 243, 226, 231, 209, 246, 230, 218, 195, 220, 268, 294, 355, 383, 537, 711, 1090, 1292, 1852, 2491, 3215, 4020, 4754, 5451, 5829, 6313, 6221, 6038, 5563, 4856, 4140, 3341, 2775, 1960, 1495, 1077, 778, 618, 478, 388, 288, 278, 243, 212, 202, 201, 209, 204, 204, 191, 197, 193, 207, 208, 208, 175, 214, 226, 248, 240, 295, 309, 427, 479, 571, 764, 951, 1112, 1397, 1642, 1786, 2021, 2249, 2212, 2138, 2053, 1799, 1599, 1334, 1161, 911, 728, 521, 394, 349, 246, 198, 171, 180, 161, 166, 164, 167, 137, 139, 153, 144, 143, 141, 132, 165, 161, 173, 140, 159, 151, 145, 147, 156, 167, 162, 188, 150, 255, 210, 224, 277, 290, 300, 274, 296, 354, 283, 257, 266, 245, 230, 209, 183, 183, 166, 138, 144, 175, 146, 155, 131, 125, 136, 146, 155, 126, 138, 136, 145, 160, 128, 137, 175, 138, 135, 146, 130, 138, 150, 161, 133, 131, 150, 128, 128, 145, 141, 132, 121, 162, 150, 134, 138, 150, 132, 135, 141, 146, 129, 167, 147, 143, 144, 144, 146, 135, 130, 154, 133, 130, 142, 133, 123, 124, 144, 126, 139, 131, 132, 136, 133, 147, 156, 151, 144, 121, 150, 133, 135, 127, 130, 154, 114, 115, 125, 125, 117, 124, 135, 124, 137, 141, 150, 119, 140, 140, 142, 151, 136, 150, 119, 142, 142, 146, 133, 147, 106, 137, 120, 130, 151, 135, 116, 148, 127, 146, 124, 117, 152, 150, 145, 141, 139, 128, 127, 118, 140, 153, 153, 161, 171, 164, 168, 160, 187, 196, 224, 199, 240, 255, 246, 233, 242, 227, 246, 236, 251, 205, 188, 186, 213, 185, 192, 195, 185, 181, 161, 148, 135, 146, 148, 161, 162, 156, 137, 183, 156, 172, 151, 162, 169, 166, 171, 154, 172, 148, 157, 154, 165, 164, 178, 166, 176, 202, 185, 175, 148, 167, 157, 165, 179, 167, 179, 193, 175, 172, 167, 186, 168, 171, 178, 199, 171, 174, 167, 195, 155, 175, 199, 189, 184, 179, 202, 169, 200, 171, 158, 199, 182, 220, 211, 191, 191, 193, 205, 200, 191, 187, 192, 194, 203, 198, 224, 189, 193, 205, 221, 205, 195, 207, 199, 239, 219, 223, 254, 248, 244, 248, 240, 276, 238, 257, 268, 261, 237, 252, 250, 252, 261, 244, 280, 211, 274, 252, 260, 269, 246, 251, 271, 262, 282, 253, 291, 286, 302, 309, 273, 285, 321, 290, 308, 327, 290, 303, 341, 318, 350, 368, 367, 377, 417, 373, 339, 346, 363, 367, 380, 380, 385, 415, 397, 424, 382, 389, 436, 459, 437, 436, 407, 395, 430, 427, 440, 401, 396, 449, 388, 415, 457, 436, 415, 460, 468, 494, 497, 472, 476, 453, 490, 505, 550, 539, 501, 549, 516, 526, 548, 570, 557, 587, 600, 568, 605, 613, 623, 661, 625, 625, 658, 641, 649, 692, 661, 711, 696, 706, 769, 705, 759, 759, 743, 773, 792, 764, 849, 801, 860, 868, 839, 871, 895, 877, 873, 900, 971, 981, 897, 996, 1037, 1041, 1004, 996, 1015, 1036, 1088, 1110, 1143, 1176, 1151, 1183, 1235, 1171, 1245, 1244, 1290, 1254, 1319, 1297, 1463, 1354, 1300, 1405, 1425, 1512, 1458, 1539, 1565, 1514, 1564, 1619, 1644, 1593, 1681, 1698, 1703, 1732, 1669, 1838, 1839, 1883, 1905, 1839, 1946, 1956, 1966, 2014, 2054, 2017, 2162, 2168, 2257, 2284, 2216, 2183, 2360, 2324, 2341, 2429, 2323, 2468, 2446, 2530, 2614, 2665, 2610, 2692, 2800, 2801, 2708, 2825, 2927, 2961, 2945, 2964, 3100, 3087, 3172, 3206, 3285, 3296, 3282, 3436, 3390, 3340, 3541, 3621, 3602, 3719, 3756, 3866, 3924, 4014, 3964, 4084, 4087, 4198, 4085, 4229, 4354, 4290, 4505, 4521, 4555, 4752, 4540, 4761, 4786, 4881, 4932, 4944, 5100, 5132, 5282, 5279, 5327, 5375, 5484, 5602, 5627, 5840, 5824, 5993, 6079, 6053, 6150, 6089, 6334, 6345, 6542, 6574, 6611, 6697, 6914, 6977, 7027, 7291, 7259, 7452, 7682, 7535, 7719, 7929, 7990, 8027, 8100, 8273, 8446, 8653, 8547, 8618, 8821, 8999, 9100, 9282, 9236, 9492, 9769, 9796, 9998, 9915, 10276, 10318, 10676, 10641, 10905, 11072, 11464, 11130, 11396, 11578, 11555, 12139, 12095, 12358, 12499, 12740, 12923, 13026, 13331, 13212, 13758, 13617, 13968, 14070, 14408, 14841, 14922, 14834, 15335, 15648, 15786, 15993, 16229, 16389, 16594, 16678, 17276, 17198, 17715, 17438, 18032, 18305, 18385, 18642, 18735, 19284, 19292, 19410, 19907, 20014, 20753, 20507, 21043, 21125, 21334, 21480, 22258, 21930, 22338, 22721, 22756, 23084, 23317, 23670, 23636, 24037, 24457, 24442, 24825, 24975, 25280, 25306, 25916, 25700, 26228, 26691, 26838, 27188, 27314, 27858, 28046, 28537, 28312, 28858, 29041, 28791, 29743, 29594, 30549, 30281, 30857, 30875, 31012, 31537, 31406, 32018, 31820, 32551, 32610, 32724, 32880, 33551, 33667, 33970, 34270, 34507, 35105, 34911, 35161, 35585, 35745, 36411, 35877, 36819, 36651, 36801, 36778, 37187, 37966, 37617, 37914, 37935, 38237, 38750, 38365, 38995, 39157, 39628, 39480, 39504, 39924, 40233, 40019, 40443, 40808, 40458, 41227, 40552, 41837, 41199, 41397, 41756, 41515, 41795, 41872, 42565, 42328, 42767, 42530, 42545, 43160, 43128, 43358, 42830, 43539, 43033, 43548, 43359, 43288, 44092, 44099, 43907, 44137, 44332, 43790, 44528, 44059, 44139, 44444, 44187, 44577, 44300, 44691, 44308, 44831, 45232, 44373, 44958, 44630, 45019, 44694, 44635, 44541, 44441, 44625, 44264, 45137, 44490, 44323, 44321, 44510, 44538, 44245, 44492, 43883, 44557, 44153, 44335, 43602, 43987, 43718, 43838, 43998, 43258, 43909, 43325, 43356, 43033, 42878, 43954, 42806, 43075, 42616, 42610, 41897, 42578, 41798, 42422, 42262, 41813, 41912, 41303, 41353, 41279, 41032, 41029, 40318, 41040, 40264, 40616, 40095, 40006, 39827, 39782, 39646, 39169, 39205, 38922, 38990, 38548, 38585, 38503, 37987, 38221, 37503, 37726, 37269, 37118, 36717, 36769, 36365, 36076, 36556, 35919, 36050, 35375, 35455, 35192, 34479, 34761, 34201, 34218, 33712, 33702, 33389, 33524, 32785, 32967, 32812, 32417, 32239, 31858, 31646, 31312, 31374, 31128, 30523, 30707, 30045, 30309, 29750, 29748, 29243, 29035, 28774, 28482, 28735, 27738, 28020, 27494, 27430, 27193, 26635, 26844, 26251, 26317, 25829, 25793, 25199, 25007, 24536, 24496, 24180, 23854, 23975, 23659, 23492, 22709, 22640, 22832, 22442, 21936, 21802, 21814, 21457, 21105, 20788, 20388, 20659, 20093, 20010, 19536, 19990, 19180, 19268, 18873, 18376, 18316, 17933, 17943, 17559, 17382, 16932, 16800, 16681, 16459, 16337, 15905, 15864, 15284, 15351, 15010, 14625, 14816, 14426, 14361, 14318, 13816, 13634, 13333, 13231, 12927, 12831, 12550, 12622, 12132, 12040, 11622, 11669, 11704, 11325, 11238, 10911, 10884, 10658, 10493, 10214, 9995, 9929, 9622, 9575, 9197, 9103, 9191, 8798, 8754, 8521, 8555, 8222, 8282, 7984, 7920, 7698, 7566, 7351, 7177, 7161, 6953, 6777, 6642, 6729, 6460, 6371, 6308, 6011, 6014, 5749, 5631, 5479, 5360, 5326, 5173, 5178, 4909, 4927, 4755, 4553, 4528, 4400, 4410, 4324, 4209, 4135, 4084, 3923, 3806, 3735, 3706, 3604, 3500, 3458, 3357, 3213, 3209, 3165, 3012, 2971, 2873, 2838, 2664, 2610, 2628, 2514, 2472, 2449, 2370, 2263, 2220, 2186, 2118, 2203, 2046, 1952, 1987, 1875, 1855, 1767, 1715, 1759, 1651, 1625, 1567, 1566, 1465, 1453, 1415, 1360, 1381, 1371, 1286, 1309, 1266, 1239, 1184, 1157, 1154, 1104, 1083, 1090, 944, 1066, 971, 966, 950, 940, 936, 920, 825, 853, 866, 810, 862, 808, 796, 795, 770, 830, 745, 717, 724, 774, 746, 683, 676, 690, 668, 707, 668, 668, 644, 641, 650, 611, 646, 619, 614, 622, 631, 575, 622, 595, 610, 645, 671, 590, 625, 611, 620, 602, 616, 575, 576, 620, 579, 565, 596, 591, 641, 634, 604, 632, 629, 603]
Cd_purafication_1 = Cd_purafication_1[299:1400]

# 2.再对数据进行Savitzky-Golay平滑滤波，使其平整并去噪
cali_data_2 = []
for value in cali_data_1.values():
    cali_data_2.append(SG(value, w=7, p=3))
# print('cali2', cali_data_2)

Cd_2 = []
for value in Cd_1.values():
    Cd_2.append(SG(value, w=7, p=3))
# print('Cd2', len(Cd_2))

Cd_purafication_2 = SG(Cd_purafication_1, w=7, p=3)

# 3.先再对数据应用基线矫正算法
baseline_fitter = Baseline(Channel, check_finite=False)
bkg_1 = []
for value in Cd_2:
    bkg1 = baseline_fitter.aspls(value, lam=10000000, max_iter=200, weights=None)[0]
    bkg_1.append(bkg1)
Cd_purafication_bkg_1 = baseline_fitter.aspls(Cd_purafication_2, lam=10000000, max_iter=200, weights=None, alpha=None)[0]

# 去除负峰
bkg_2 = []
for value1, value2 in zip(bkg_1, Cd_2):
    bkg2 = refine(value1, value2)
    bkg_2.append(bkg2)
Cd_purafication_bkg_2 = refine(Cd_purafication_bkg_1, Cd_purafication_2)

bkg_3 = []
for value in bkg_2:
    bkg3 = baseline_fitter.aspls(value, lam=4000000, max_iter=200, weights=None)[0]
    bkg_3.append(bkg3)
Cd_purafication_bkg_3 = baseline_fitter.aspls(Cd_purafication_2, lam=5000000, max_iter=200, weights=None, alpha=None)[0]

Cd_3_1 = []
for x1, x2 in zip(Cd_2, bkg_1):
    s_1 = np.arange(len(Cd_2[0]), dtype=np.float64)
    for n in range(0, len(Cd_2[0])):
        s_1[n] = x1[n] - x2[n]
    Cd_3_1.append(s_1)
Cd_purafication_3_1 = Cd_purafication_2-Cd_purafication_bkg_1


# 去除负峰后
Cd_3_2 = []
for x1, x2 in zip(bkg_2, bkg_3):
    s_2 = np.arange(len(Cd_2[0]), dtype=np.float64)
    for n in range(0, len(Cd_2[0])):
        s_2[n] = x1[n] - x2[n]
    Cd_3_2.append(s_2)
Cd_purafication_3_2 = Cd_purafication_bkg_2-Cd_purafication_bkg_3

# print(np.array(Cd_3_2))

# # 校正数据的基线校正
# cali_bkg_1 = []
# for value in cali_data_2:
#     cali_bkg1 = baseline_fitter.aspls(value, lam=1000000, max_iter=200, weights=None, alpha=None)[0]
#     cali_bkg_1.append(cali_bkg1)
#
# cali_bkg_2 = []
# for value1, value2 in zip(cali_bkg_1, cali_data_2):
#     cali_bkg2 = refine(value1, value2)
#     cali_bkg_2.append(cali_bkg2)
#
# cali_bkg_3 = []
# for value in cali_bkg_2:
#     cali_bkg3 = baseline_fitter.aspls(value, lam=400000, max_iter=200, weights=None, alpha=None)[0]
#     cali_bkg_3.append(cali_bkg3)
#
# cali_data_3_1 = []
# for x1, x2 in zip(cali_data_2, cali_bkg_1):
#     s_1 = np.arange(len(cali_data_2[0]), dtype=np.float64)
#     for n in range(0, len(cali_data_2[0])):
#         s_1[n] = x1[n] - x2[n]
#     cali_data_3_1.append(s_1)
#
# # 去除负峰后
# cali_data_3_2 = []
# for x1, x2 in zip(cali_data_2, cali_bkg_3):
#     s_2 = np.arange(len(cali_data_2[0]), dtype=np.float64)
#     for n in range(0, len(cali_data_2[0])):
#         s_2[n] = x1[n] - x2[n]
#     cali_data_3_2.append(s_2)
# # print('cali3', cali_data_3_2)

# 二阶导数寻峰
# Zn150_D2_1 = D2(Zn150_3_2)
# Zn150_D2_2 = SG(Zn150_D2_1.T, w=7, p=3).T
# Cd070_3_D2 = D2(Cd070_3)
# Find valleys
# order:两侧使用多少点进行比较
# valley_indexes = argrelextrema(Zn150_D2_2, np.less, order=15)
# valley_indexes = valley_indexes[0]

# 使用scipy库函数find_peaks寻峰
# peaks = []
# for value in Cd_3_2:
#     peak, _ = find_peaks(value, prominence=5)  # 太牛了！
#     peaks.append(peak)

# peak1 = [0.0197787 * x + 5.9050229 for x in peaks]

Cd_Ka = 871

# 未应用图像处理和蒙特卡洛求拟合面积，即只取佳谱默认的峰位左右5道内数据作为面积
# area = []
# for i in Cd_3_2:
#     sum1 = sum(i[866:877])
#     area.append(sum1)
# print("area", len(area))

# Zn_concentration = np.array([[1000 * float(key[3:5]) for key in Cd_1]])

Cd_concentration = np.array([[1000 * float(key[9:]) for key in Cd_1]])
Cd_concentration = Cd_concentration.ravel()

# concentration = np.vstack((Zn_concentration, Cd_concentration)).T
# print("znconcentration", Zn_concentration)
# print("CDconcentration", Cd_concentration)
# print("concentration", concentration.shape)

# _, w, b = calc_curve(np.array(area), np.array(concentration))
# print("RMSE", rmse)
# print('r2', r2)

# # 利用校正数据求出Cd的校正曲线
# cali_concentration = [float(key[3:-4]) for key in cali_data_1]
# cali_concentration = [i * 1000 for i in cali_concentration]
# # print('cali_concen', cali_concentration)
# cali_area = []
# for i in cali_data_3_2:
#     sum2 = sum(i[861:882])
#     cali_area.append(sum2)
# print('cali_area', cali_area)
# # cali_result = simulated_annealing(cali_area, cali_concentration)
# cali_curve, cali_w, cali_b = calc_curve(np.array(cali_area), np.array(cali_concentration))
# # print('cali_curve', cali_curve)
# # rmse1, r2_1 = get_rmse_r2(cali_concentration, cali_curve)
# # print("RMSE,r2_1", rmse1,r2_1)

# 划分训练集和测试集

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split \
    (np.array(Cd_3_2), Cd_concentration, train_size=0.7, random_state=0)

# print('X_train', X_train)
# print('X_test', X_test)
# print('y_train', y_train)
# print('y_test', y_test)

scaler = StandardScaler()
train_x = scaler.fit_transform(X_train)
test_x = scaler.transform(X_test)

# 特征选择
from auswahl import RandomFrog

# rmsecv_array = []
# rmsep_array = []
# r2_array = []

n_iterations = 1000
n_features_to_select = 13

# n_iterations = [500, 1000, 1500, 2000, 2500, 3000]
# n_features_to_select = [3, 5, 7, 9, 11, 13]

rf = RandomFrog(n_features_to_select=n_features_to_select, n_iterations=n_iterations, n_jobs=5, random_state=7331)
new_X_train = rf.fit_transform(train_x, y_train)
new_X_test = rf.transform(test_x)
frq = rf.frequencies_
#
# # print('colors', colors)
# # print('mask', rf.get_support())
# # print('rf.frequencies_', frq)
# frequencies = sorted(rf.frequencies_, reverse=True)[0:n_features_to_select]
# # print(frequencies)
feature_names = rf.get_feature_names_out()
feature_names = [int(x[1:])+1 for x in feature_names]
print(feature_names)
Energy12 = [0.0197787 * (x+300) - 0.0483658 for x in feature_names]  # 能量，光谱的横轴
print(Energy12)
#
# # 选出概率前n_features_to_select个特征组成新训练集
# new_X_train = np.zeros((train_x.shape[0], n_features_to_select))
# for i in range(train_x.shape[0]):
#     for j in range(n_features_to_select):
#         new_X_train[i][j] = (train_x[i][feature_names[j]-1])
# print(type(new_X_train))
#
# new_X_test = np.zeros((test_x.shape[0], n_features_to_select))
# for i in range(test_x.shape[0]):
#     for j in range(n_features_to_select):
#         new_X_test[i][j] = (test_x[i][feature_names[j]-1])
# # print(new_X_test)

# from sklearn.decomposition import PCA
#
# n_components = 40
# pca = PCA(n_components=n_components, random_state=7331)
# new_X_train = pca.fit_transform(train_x)
# print(pca.explained_variance_ratio_.sum())
# new_X_test = pca.transform(test_x)
#
# print(train_x.shape[1])
# print(new_X_train.shape[1])



# feature_names = pca.get_feature_names_out()
# feature_names = [int(x[1:])+1 for x in feature_names]

# 选出概率前n_features_to_select个特征组成新训练集
# new_X_train = np.zeros((train_x.shape[0], n_components))
# for i in range(train_x.shape[0]):
#     for j in range(n_components):
#         new_X_train[i][j] = (train_x[i][feature_names[j]-1])
# print(type(new_X_train))
#
# new_X_test = np.zeros((test_x.shape[0], n_components))
# for i in range(test_x.shape[0]):
#     for j in range(n_components):
#         new_X_test[i][j] = (test_x[i][feature_names[j]-1])
# # print(new_X_test)

# ew_Cd_purification = Cd_purafication_3_2[feature_names]
# print('new_Cd_purification', type(new_Cd_purification))
# from auswahl import IntervalRandomFrog
#
# # np.random.seed(1337)
# X = train_x
# y = y_train
#
# n_iterations = 1000
# n_intervals_to_select = 15
# interval_width = 2
# irf = IntervalRandomFrog(n_intervals_to_select=n_intervals_to_select,
#                          interval_width=interval_width,
#                          n_iterations=n_iterations,
#                          n_jobs=5,
#                          random_state=7331)
# irf.fit(X, y)
#
# # print('rf.frequencies_', frq)
# frequencies = sorted(irf.frequencies_, reverse=True)[0:n_intervals_to_select * interval_width]
# # print(frequencies)
# feature_names = irf.get_feature_names_out()
# feature_names = [int(x[1:])+1 for x in feature_names]
# # print(feature_names)
# Energy12 = [0.0197787 * (x+300) - 0.0483658 for x in feature_names]  # 能量，光谱的横轴
# # print(Energy12)
#
# # 选出概率前n_intervals_to_select * interval_width个特征组成新训练集
# new_X_train = np.zeros((train_x.shape[0], n_intervals_to_select * interval_width))
# for i in range(train_x.shape[0]):
#     for j in range(n_intervals_to_select * interval_width):
#         new_X_train[i][j] = (train_x[i][feature_names[j]])
# # print(new_X_train.shape)
#
# new_X_test = np.zeros((test_x.shape[0], n_intervals_to_select * interval_width))
# for i in range(test_x.shape[0]):
#     for j in range(n_intervals_to_select * interval_width):
#         new_X_test[i][j] = (test_x[i][feature_names[j]])
# # print(new_X_test)

# 利用校正数据求出Cd的校正曲线
# # cali_concentration = [float(key[3:-4]) for key in cali_data_1]
# cali_concentration = y_train
# print('cali_concen', cali_concentration)
# cali_area = []
# for i in X_train:
#     sum2 = sum(i[861:882])
#     cali_area.append(sum2)
# print('cali_area', cali_area)
# cali_result = simulated_annealing(cali_area, cali_concentration)
# cali_curve, cali_w, cali_b = calc_curve(np.array(cali_area), np.array(cali_concentration))
# # print('cali_curve', cali_curve)
# print(cali_w, cali_b)
# # rmse1, r2_1 = get_rmse_r2(cali_concentration, cali_curve)
# # print("RMSE,r2_1", rmse1,r2_1)
#

parameters = {
    "kernel": ["rbf"],
    "sigma": [0.01, 0.1,1,10,100,1000],
    "gamma": [1,10,100,1000,10000,100000]
    }

grid = GridSearchCV(LSSVMRegression(), parameters, cv=5, verbose=1)

grid.fit(new_X_train, y_train)
print('best params', grid.best_params_)
best_estimator = grid.best_estimator_

y_pred_cv = cross_val_predict(best_estimator, new_X_train, y_train, cv=5)
y_pred = best_estimator.predict(new_X_test)
# print("new_X_test", new_X_test)
# Cd_purafication_pred = best_estimator.predict(new_Cd_purification.reshape(1, -1))

# print('y_pred', y_pred)
# print('Cd_purafication_pred', Cd_purafication_pred)

# 计算评价指标（rmsecv,rmsep,r2）
rmsecv = sqrt(np.mean(cross_val_score(best_estimator, new_X_train, y_train, cv=5)))
rmsep = sqrt(mean_squared_error(y_test, y_pred))
r_2_p = r2_score(y_test, y_pred)

are = average_relative_error(y_test, y_pred)
mre = maximum_relative_error(y_test, y_pred)
print(f"交叉验证均方根误差 (RMSECV): {rmsecv:.4f}")
print(f"预测均方根误差 (RMSEP): {rmsep:.4f}")
print(f"相关系数 (R2): {r_2_p:.4f}")
print(f"平均相对误差 (ARE): {are*100:.4f}%")
print(f"最大相对误差 (MRE): {mre*100:.4f}%")
print('--------------------------------')
# print('测试集数据', y_test)
# print('预测数据', y_pred)
print('--------------------------------')

# rmsecv_array.append(rmsecv)
# rmsep_array.append(rmsep)
# r2_array.append(r_2_p)

# print('rmsecv_array, rmsep_array, r2_array', rmsecv_array, rmsep_array, r2_array)
# sns.set(style="whitegrid")

# 输出测试图像部分
font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 16,
         }
plt.rcParams['font.family'] = ['SimHei']  # 设置中文显示字体为黑体

fig, ax1 = plt.subplots()

for data in Cd_3_2:
    ax1.plot(Energy, data, color='blue')
ax1.set_xlabel('Energy(keV)', fontdict=font1)
ax1.set_ylabel('Intensity(cps)', fontdict=font1, color='blue')
ax1.set_ylim(0, 20000)
ax1.tick_params('y', colors='blue')
#
ax2 = ax1.twinx()
colors = np.full(train_x.shape[1], fill_value='dodgerblue')
colors[rf.get_support()] = 'darkgreen'
ax2.bar(Energy, rf.frequencies_ / n_iterations * 100, alpha=0.3, color=colors, width=0.1)
ax2.set_ylabel('Relative Frequency(%)', color='seagreen', fontdict=font1)
ax2.set_ylim(0, 100)  # 设置右侧y轴的范围
ax2.tick_params('y', colors='green')

# plt.title('Spectra & features selected by Random Frog', fontdict=font1)

plt.legend()
plt.axis('auto')
plt.axhline(y=0, linestyle='--', linewidth=1)

plt.show()

# plt.bar(x=np.arange(train_x.shape[1])+1, height=rf.frequencies_ / n_iterations * 100, color=colors)
# plt.scatter(feature_names, rf.frequencies_[feature_names]/10, label='选择的特征')

# plt.rcParams['axes.unicode_minus'] = False    # 解决保存图像时负号'-'显示为方块的问题

# plt.title('预处理前后光谱对比')
# fig1 = plt.figure(1)
# plt.xlabel('Channel', fontdict=font1)
# plt.ylabel('Intensity(cps)', fontdict=font1)
#
# for data in Cd_1.values():
#     plt.plot(Channel, data, '-')
# plt.plot(Energy, Cd_purafication_1, '--', label='原光谱')


# fig2 = plt.figure(2)
# plt.xlabel('Energy(keV)', fontdict=font1)
# plt.ylabel('Intensity(cps)', fontdict=font1)
# for data in Cd_2:
#     plt.plot(Energy, data, 'g-')
# # plt.plot(Energy, Cd_2[0], '--', label='原光谱')
# plt.plot(Energy, Cd_purafication_2, '-', label='SG滤波后的的光谱')


# for x in bkg_1:
#     plt.plot(Energy, x, '--', color='r')
# for x in bkg_2:
#     plt.plot(Channel, x, '--', color='r')
# for x in bkg_3:
#     plt.plot(Channel, x, '--', color='r')

# plt.plot(Energy,bkg_3_2, '--', label='aspls_b2')
# plt.plot(Energy,bkg_3, '--', label='aspls-b3')

# for data in Cd_3_1:
#     plt.plot(Energy, data, 'b-')
# plt.plot(Energy, Cd_3_1[0], '-', label='预处理后的光谱')
# plt.plot(Energy, Cd_purafication_3_1, '-', )
# plt.plot(Energy, Cd_purafication_3_2, '-', label='预处理后的光谱')
# for data in Cd_3_2:
#     plt.plot(Energy, data, 'g-')
# for data in Cd_3_2:
#     plt.plot(Energy, data, '-')
# plt.plot(Energy, Cd_3_2[0], '-', label='消除负峰之后的光谱')
# plt.plot(Channel, Cd_3_2[0], '-', color='black')

# plt.plot(Channel2,Zn150_D2_1,'-',label='second_derivative_Zn_1')
# plt.plot(Channel2,Zn150_D2_2,'-',label='second_derivative_Zn_2')

# valley_x_1 = valley_indexes + 301
# valley_y = Zn150_D2_2[valley_indexes]
# valley_x_2 = valley_indexes + 301
# Peak = Zn150_3_2[valley_indexes + 1]
# plt.scatter(valley_x_1, valley_y, marker='o', color='green', label="Valleys")
# plt.scatter(valley_x_2, Peak, marker='o', color='red', label="Peaks")
# for value, peak in zip(Cd_2, peaks):
#     plt.plot(peak + 300, value[peak], 'o', color='red')

# plt.legend()
# plt.axis('auto')
# plt.axhline(y=0, linestyle='--', linewidth=1)
# #
# plt.show()

# plt.plot(Channel,Cd070_1,'-',label='log_Cd070')
# plt.plot(Channel1,Cd070_2,'-',label='SG_Cd070')
# plt.plot(Channel1,bkg_Cd, '--', label='aspls_Cd')
# plt.plot(Channel1,Cd070_3, '-', label='after_aspls_SG_log_Cd070')
# plt.plot(Channel2,Cd070_3_D2,'-',label='second_derivative_Cd')

# RF特征选择相对概率图(RF)
# plt.figure(2)
# plt.bar(x=np.arange(train_x.shape[1])+1, height=rf.frequencies_ / n_iterations * 100, color=colors)
# # plt.scatter(feature_names, rf.frequencies_[feature_names]/10, label='选择的特征')
#
# plt.title('RF算法特征选择相对概率')
# plt.xlabel('特征序号')
# plt.ylabel('相对概率(%)')
# plt.legend()
# plt.show()

# RF特征选择相对概率图(IRF)
# plt.figure(2)
# idx = np.arange(len(irf.frequencies_))
# plt.plot(idx, irf.frequencies_ / n_iterations, marker='.', zorder=3)
# plt.hlines(y=irf.frequencies_ / n_iterations,
#            xmin=idx,
#            xmax=idx + irf.interval_width - 1,
#            alpha=0.5,
#            zorder=1)
#
# interval_starts = np.argwhere(np.diff(irf.get_support().astype(int)) > 0) + 1
# plt.hlines(y=irf.frequencies_[interval_starts] / n_iterations,
#            xmin=interval_starts,
#            xmax=interval_starts + irf.interval_width - 1,
#            colors='C01',
#            zorder=2)
#
# plt.grid(axis='y')
#
# plt.xlabel('Feature')
# plt.ylabel('Relative Frequency')
# plt.legend(['Frequency', 'Interval', 'Selected Intervals'])
#
# plt.show()


# plt.figure(3)
# plt.plot(range(1, 25), y_test, color='g')
# plt.scatter(range(1, 25), y_test, marker='^', color='g', label='Cd真实值')
# plt.plot(range(1, 25), y_pred, color='r')
# plt.scatter(range(1, 25), y_pred, marker='s', color='r', label='Cd预测值')
#
# # plt.plot(concentration, pred_area, '-', color='b', label='prediction_value')
# plt.title('预测结果')
# plt.xlabel('样本序数')
# plt.ylabel('Cd浓度(mg/L)')
# # plt.annotate('R^2 = {:.6f}'.format(r2), xy=(500, 3000), xytext=None)

# plt.axis('auto')
# plt.legend()
# plt.show()
# 示例数

# # 预测区间（假设）
# lower_95 = np.mean(y_pred) - np.std(y_pred)*1.96
# upper_95 = np.mean(y_pred) + np.std(y_pred)*1.96
# print(lower_95)
#
# lower_90 = np.mean(y_pred) - np.std(y_pred)*1.645
# upper_90 = np.mean(y_pred) + np.std(y_pred)*1.645
#
# lower_99 = np.mean(y_pred) - np.std(y_pred)*2.576
# upper_99 = np.mean(y_pred) + np.std(y_pred)*2.576
#
# # 画图
# plt.figure(figsize=(8, 12))
#
# for i in range(len(y_test)):
#     # plt.plot([lower_95[i], upper_95[i]], [i + 1, i + 1], color='lightblue', linewidth=10, alpha=0.8)
#     # plt.plot([lower_90[i], upper_90[i]], [i + 1, i + 1], color='deepskyblue', linewidth=7, alpha=0.8)
#     # plt.plot([lower_99[i], upper_99[i]], [i + 1, i + 1], color='dodgerblue', linewidth=5, alpha=0.8)
#     plt.scatter(i + 1, y_pred[i], color='blue', s=50)
#     plt.scatter(i + 1, y_test[i],  edgecolor='red', facecolor='none', s=70, linewidth=2)
#
# plt.ylabel('Predicted Concentration(mg/L)', fontsize=14)
# plt.xlabel('Number of test samples', fontsize=14)
# plt.xticks(np.arange(1, len(y_test) + 1))
# plt.yticks(np.arange(0, 110, 20), ['0', '20', '40', '60', '80', '100'])
# plt.grid(True, axis='y', linestyle=':', alpha=0.7)
#
# plt.legend(['Predicted value', 'Actual value'],
#            loc='upper right', fontsize=12)
#
# plt.show()
