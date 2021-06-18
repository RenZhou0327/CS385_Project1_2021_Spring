import pickle

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

from matplotlib import cbook
from matplotlib import cm
from matplotlib.colors import LightSource
import matplotlib.pyplot as plt
import numpy as np

# # Load and format data
# with cbook.get_sample_data('jacksboro_fault_dem.npz') as file, \
#      np.load(file) as dem:
#     z = dem['elevation']
#     nrows, ncols = z.shape
#     x = np.linspace(dem['xmin'], dem['xmax'], ncols)
#     y = np.linspace(dem['ymin'], dem['ymax'], nrows)
#     x, y = np.meshgrid(x, y)
#
# region = np.s_[5:50, 5:50]
# x, y, z = x[region], y[region], z[region]
# print(x.shape, y.shape, z.shape)
# print(x)
# exit(0)
x = []
y = []
for i in range(0, 40):
    x.append([])
    y.append([])
    for j in range(0, 40):
        x[i].append(i)
        y[i].append(j)
x = np.array(x)
y = np.array(y)
print(x.shape)
print(y.shape)
f = open("Lasso_w_0001.bin", "rb")
data = pickle.load(f)
f.close()
data = data[:1600]
data = data.cpu().numpy()
z = data.reshape((40, 40))
# print(data.shape)
# exit(0)
# print(x)
# print(y)
# exit(0)
# Set up plot
fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))

ls = LightSource(270, 45)
# To use a custom hillshading mode, override the built-in shading and pass
# in the rgb colors of the shaded surface calculated from "shade".
rgb = ls.shade(z, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=rgb,
                       linewidth=0, antialiased=False, shade=False)
plt.tight_layout()
plt.savefig("Lam_Lasso0001.png")
plt.show()