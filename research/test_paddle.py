import paddle

print(paddle.__version__)
print(paddle.device.is_compiled_with_cuda())
print(paddle.device.get_device())
