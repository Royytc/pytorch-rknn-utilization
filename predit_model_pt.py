import torch
import onnxruntime as rt
import numpy
input_test_data = torch.tensor([[15]]).float()
net = torch.load('linear_model.pt')
prediction = net(input_test_data)
print(prediction)

sess = rt.InferenceSession('linear_model.onnx')
input_name = sess.get_inputs()[0].name
print(input_name)
pred_onx = sess.run(None, {input_name:input_test_data.numpy()})[0]
print(pred_onx)
