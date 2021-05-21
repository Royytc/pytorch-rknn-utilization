from rknnlite.api import RKNNLite as RKNN
import numpy as np
model='linear_model.rknn'
input_test_data = np.array([15],dtype=np.float32)
rknn=RKNN()
ret=rknn.load_rknn(path=model)
print('--> Init runtime environment')
ret = rknn.init_runtime()
if ret != 0:
    print('Init runtime environment failed')
    exit(ret)
print('done')

output=rknn.inference(inputs=[input_test_data])
print(output)
