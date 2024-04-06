import paddle
paddle.device.set_device('gpu')

s = paddle.device.cuda.Stream()

# vector * vector
x = paddle.rand([10])
y = paddle.rand([10])
with paddle.device.cuda.stream_guard(s):
    z = paddle.matmul(x, y)
print(z.shape)

# matrix * vector
x = paddle.rand([10, 5])
y = paddle.rand([5])
with paddle.device.cuda.stream_guard(s):
    z = paddle.matmul(x, y)
print(z.shape)

# batched matrix * broadcasted vector
x = paddle.rand([10, 5, 2])
y = paddle.rand([2])
with paddle.device.cuda.stream_guard(s):
    z = paddle.matmul(x, y)
print(z.shape)

# batched matrix * batched matrix
x = paddle.rand([10, 5, 2])
y = paddle.rand([10, 2, 5])
with paddle.device.cuda.stream_guard(s):
    z = paddle.matmul(x, y)
print(z.shape)

# batched matrix * broadcasted matrix
x = paddle.rand([10, 1, 5, 2])
y = paddle.rand([1, 3, 2, 5])
with paddle.device.cuda.stream_guard(s):
    z = paddle.matmul(x, y)
print(z.shape)

