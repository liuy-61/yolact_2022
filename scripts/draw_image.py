import matplotlib.pyplot as plt
import numpy as np

# x = np.linspace(-1, 1, 50)
# y = 2*x + 1

# x = np.array([1, 2, 3, 4, 5])
# base_y = np.array([34.57, 34.57, 34.57, 34.57, 34.57])
# loss_y = np.array([34.64, 34.57, 34.63, 34.82, 34.75])
# double_loss_y = np.array([34.75, 34.79, 34.81, 34.87, 34.79])
# confidence_y = np.array([35.21, 35.38, 35.23, 35.18, 35.13])
# KD_y = np.array([35.16, 35.19, 35.18, 35.17, 35.04 ])
# plt.figure()
# plt.plot(x, base_y)
# plt.plot(x, loss_y)
# plt.plot(x, double_loss_y)
# plt.plot(x, confidence_y)
# plt.plot(x, KD_y)
# plt.xlabel('k')
# plt.ylabel('BoxAp')
# plt.show()


# plt.rcParams['font.sans-serif'] = ['SimHei'] # 步骤一（替换sans-serif字体）
# plt.rcParams['axes.unicode_minus'] = False   # 步骤二（解决坐标轴负数的负号显示问题）
#
x = np.array([10, 20, 30])
base_y = np.array([68.71, 65.17, 64.81])
VAAL_y = np.array([69.97, 69.81, 69.06])
core_set_y = np.array([69.71, 69.64, 68.96])

plt.figure()

plt.plot(x, base_y)
plt.plot(x, VAAL_y)
plt.plot(x, core_set_y)

plt.legend(['random', 'VAAL', 'core-set'])

plt.xlabel('k')
plt.ylabel('BoxAp')
plt.locator_params(nbins=3)
plt.show()

# plt.rcParams['font.sans-serif'] = ['SimHei'] # 步骤一（替换sans-serif字体）
# plt.rcParams['axes.unicode_minus'] = False   # 步骤二（解决坐标轴负数的负号显示问题）
#
# x = np.array([10, 20, 30])
# base_y = np.array([61.46, 57.97, 55.69])
# VAAL_y = np.array([62.93, 62.79, 62.57])
# core_set_y = np.array([62.64, 62.62, 62.38])
#
# plt.figure()
#
# plt.plot(x, base_y)
# plt.plot(x, VAAL_y)
# plt.plot(x, core_set_y)
#
# plt.legend(['random', 'VAAL', 'core-set'])
#
# plt.xlabel('k')
# plt.ylabel('MaskAp')
# plt.locator_params(nbins=3)
# plt.show()


