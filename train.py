import torch
from  pytorch_linear_model import net
x=torch.unsqueeze(torch.arange(-100,100),dim=1).float()
y=x.pow(2)
def save():
    optimizer=torch.optim.Adam(net.parameters(),lr=0.1)
    loss_func=torch.nn.MSELoss()
    for t in range(2000):
        prediction=net(x)
        loss=loss_func(prediction,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.save(net,'linear_model.pt')
    input_shape = (1)  # 输入数据
    batch_size=1

    input_data_shape = torch.randn(batch_size, input_shape)

    torch.onnx.export(net, input_data_shape, "linear_model.onnx", verbose=True)
save()