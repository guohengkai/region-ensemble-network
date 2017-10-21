"""
Caculate yaw and pitch angle of hand poses for MSRA dataset.

Author: Hengkai Guo, Xinghao Chen

"""
import numpy as np
import matplotlib.pyplot as plt
import util

show_validate_figure = 0

# load hand poses
labels = np.loadtxt('../labels/msra_test_label.txt')
labels = np.reshape(labels, (-1, 21, 3))

# first convert uvd to xyz
params = util.get_param('msra')
labels = util.pixel2world(labels, *params)
                   
# wrist, mcp for index, middle, ring, little, thumb
labels = labels[:, [0, 1, 5, 9, 13, 17], :]

''' 
The palm coordinate frame has its origin at wrist, positive Y axis
pointing to the middle finger root and positive Z axis pointing
outwards of the palm plane.

Sun et al. "Cascaded Hand Pose Regression", CVPR 2015
'''

# y axis
y = labels[:, 2, :] - labels[:, 0, :]
# z axis
little = labels[:, 4, :] - labels[:, 0, :]
z = np.cross(y, little)

# normalize y axis to unit vector
row_sums = np.linalg.norm(y, axis=1)
y = y / row_sums[:, np.newaxis]

# normalize x axis to unit vector
row_sums = np.linalg.norm(z, axis=1)
z = z / row_sums[:, np.newaxis]

# calculate x axis and normalize x axis to unit vector
x = np.cross(y, z)
row_sums = np.linalg.norm(x, axis=1)
x = x / row_sums[:, np.newaxis]

# calculate pitch and yaw angles
# refs: https://afni.nimh.nih.gov/pub/dist/src/pkundu/meica.libs/nibabel/eulerangles.py
pitch = -np.arcsin(y[:, 2])
yaw = -np.arcsin(-y[:, 0] / np.cos(pitch))

pitch = pitch * 180 / np.pi
yaw = yaw * 180 / np.pi

# save results
angles = np.array([yaw, pitch])
np.savetxt('../labels/msra_angle.txt', np.transpose(angles), fmt='%0.3f %0.3f')

'''
Draw the Viewpoint (pitch and yaw) distribution of MSRA dataset.
It should look like Figure4 in Sun et al. CVPR 2015
'''  
if show_validate_figure:
    weights = np.ones_like(pitch)/float(len(pitch))
    plt.hist(pitch, bins=180, weights=weights)  # arguments are passed to np.histogram
    plt.title("pitch")
    axes = plt.gca()
    axes.set_xlim([-90, 90])
    axes.set_ylim([0, 0.06])
    plt.show()
    
    plt.figure()
    plt.hist(yaw, bins=180, weights=weights)  # arguments are passed to np.histogram
    plt.title("yaw")
    axes = plt.gca()
    axes.set_xlim([-90, 90])
    axes.set_ylim([0, 0.06])
    plt.show()
