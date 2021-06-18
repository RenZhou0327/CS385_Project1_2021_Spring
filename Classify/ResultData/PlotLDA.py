from matplotlib import pyplot as plt
import pickle
import numpy as np

f = open("LDA_Proj.bin", "rb")
x_pos_list, x_neg_list = pickle.load(f)
f.close()
for i in range(4):
    x_pos = x_pos_list[i]
    x_neg = x_neg_list[i]
    plt.figure(figsize=(8, 6))
    plt.hist(x_pos, bins=100, color="g", alpha=0.3, label="Positive")
    plt.hist(x_neg, bins=100, color="r", alpha=0.3, label="Negative")
    plt.legend(fontsize=15, loc="upper right")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel('Cnt', fontsize=15)
    plt.xlabel('Project Value', fontsize=15)
    plt.tight_layout()
    plt.savefig(f"LDA_distribute{i}.png")
    plt.close('all')