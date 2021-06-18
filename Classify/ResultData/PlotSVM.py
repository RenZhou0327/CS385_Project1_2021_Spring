from matplotlib import pyplot as plt
import pickle
import numpy as np

x = list(range(0, 7))
y1 = [0.357, 0.407, 0.679, 0.7784, 0.776, 0.7484, 0.7454]
y2 = [0.3574, 0.4064, 0.697, 0.817, 0.8202, 0.8054, 0.8058]
y3 = [0.3568, 0.4082, 0.701, 0.8256, 0.8324, 0.822, 0.8232]
fig = plt.figure(figsize=(8, 6))
plt.plot(x, y1, linewidth=2, label="C=0.1", marker="o", markersize=10)
plt.plot(x, y2, linewidth=2, label="C=0.5", marker="^", markersize=10)
plt.plot(x, y3, linewidth=2, label="C=1.0", marker="*", markersize=10)
plt.legend(fontsize=15, loc="lower right")
plt.xticks(x, [2, 3, 10, 50, 100, 500, 1000], fontsize=12)
plt.yticks(fontsize=12)
plt.ylabel('Acc(%)', fontsize=15)
plt.xlabel('Dimension', fontsize=15)
plt.tight_layout()
plt.savefig("SVM_C_Acc.png")
plt.show()
exit(0)


x = list(range(0, 7))
y1 = [0.3326, 0.3548, 0.6014, 0.7196, 0.7258, 0.7312, 0.7314]
y2 = [0.253, 0.2846, 0.5234, 0.7484, 0.7728, 0.7874, 0.791]
y3 = [0.3568, 0.4082, 0.701, 0.8256, 0.8324, 0.822, 0.8232]

fig = plt.figure(figsize=(8, 6))
plt.plot(x, y1, linewidth=2, label="Linear", marker="o", markersize=10)
plt.plot(x, y2, linewidth=2, label="Sigmoid", marker="^", markersize=10)
plt.plot(x, y3, linewidth=2, label="RBF", marker="*", markersize=10)
plt.legend(fontsize=15, loc="lower right")
plt.xticks(x, [2, 3, 10, 50, 100, 500, 1000], fontsize=12)
plt.yticks(fontsize=12)
plt.ylabel('Acc(%)', fontsize=15)
plt.xlabel('Dimension', fontsize=15)
plt.tight_layout()
plt.savefig("SVM_Kernel_Acc.png")
plt.show()
