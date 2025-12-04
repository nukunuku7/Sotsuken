import cupy as cp
print(cp.cuda.runtime.getDeviceCount())
print(cp.cuda.runtime.getDeviceProperties(0))
