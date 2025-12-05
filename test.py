import cupy
print("CuPy version:", cupy.__version__)
cupy.show_config()

import cupy.cuda.runtime as rt
try:
    print("Runtime version:", rt.runtimeGetVersion())
except Exception as e:
    print("runtimeGetVersion ERROR:", e)
