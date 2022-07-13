#%%
import kornia 
import torch
import matplotlib.pyplot as plt
import time
from math import pi
from copy import deepcopy

from get_data import xs

image = xs[0]
plt.imshow(image)
plt.show()
plt.close()

xs = xs.permute(0, -1, 1, 2)

def try_colorspace(name, function, norms = (2,2,2)):
    print("\n\nMaking {}...".format(name))
    start = time.time()
    image = function(xs)
    end = time.time()
    print(end - start, "seconds")

    print("\n{} min/max:".format(name))
    for i in range(image.shape[1]):
        print(torch.min(image[:,i,:,:]), torch.max(image[:,i,:,:]))
    
    for i in range(image.shape[1]):
        image[:,i,:,:] = image[:,i,:,:] * norms[i] - 1

    print("\n{} norm min/max:".format(name))
    for i in range(image.shape[1]):
        print(torch.min(image[:,i,:,:]), torch.max(image[:,i,:,:]))

try_colorspace("RGB", lambda image: deepcopy(image), (2,2,2))

#%%
try_colorspace("Gray", kornia.color.rgb_to_grayscale, (2,))

#%%
try_colorspace("HSV", kornia.color.rgb_to_hsv, (1/pi, 2, 2))

#%%
try_colorspace("HLS", kornia.color.rgb_to_hls, (1/pi, 2, 2))

# %%
try_colorspace("LUV", kornia.color.rgb_to_luv, (2, 2, 2)) # PROBLEMATIC

# %%
try_colorspace("LAB", kornia.color.rgb_to_lab, (2, 2, 2)) # PROBLEMATIC

# %%
try_colorspace("YCbCr", kornia.color.rgb_to_ycbcr, (2, 2, 2))

# %%
try_colorspace("YUV", kornia.color.rgb_to_yuv, (2, 2, 2)) # PROBLEMATIC

# %%
try_colorspace("XYZ", kornia.color.rgb_to_xyz, (2, 2, 2))
# %%
