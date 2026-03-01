from PIL import Image
import numpy as np

im = Image.open('gallery/cornell_box_var.png')
arr = np.array(im)
print('shape', arr.shape)

w,h = arr.shape[1], arr.shape[0]
halfw = w//2
halfh = h//2
q1=arr[:halfh,:halfw]
q2=arr[:halfh,halfw:]
q3=arr[halfh:,:halfw]
q4=arr[halfh:,halfw:]

print('max differences between quarters:')
print('q1 vs q2 max', np.max(np.abs(q1-q2)))
print('q1 vs q3 max', np.max(np.abs(q1-q3)))
print('q1 vs q4 max', np.max(np.abs(q1-q4)))

print('right seam max diff', np.max(np.abs(arr[:,halfw-1,:]-arr[:,halfw,:])))
print('bottom seam max diff', np.max(np.abs(arr[halfh-1,:,:]-arr[halfh,:,:])))
