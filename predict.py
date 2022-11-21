import mindspore
import matplotlib.pyplot as plt
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore import Tensor, set_context, PYNATIVE_MODE, dtype as mstype

set_context(mode=PYNATIVE_MODE)
# 自定义网络
# 通过设置pynative_synchronize来使算子同步执行
set_context(mode=PYNATIVE_MODE, pynative_synchronize=True)


class Net(nn.Cell):                # 网络定义

    def construct(self, a, b, x, y):
        z = mindspore.numpy.mean(((a * x + b) - y) ** 2)
        return z


def pred(dis, tem):

    x = mindspore.Tensor(dis)
    y = mindspore.Tensor(tem)

    x_train = x[:-10]
    x_test = x[-10:]
    y_train = y[:-10]
    y_test = y[-10:]
    #
    A = mindspore.Tensor(mindspore.numpy.rand(1))
    B = mindspore.Tensor(mindspore.numpy.rand(1))
    a = Tensor(A, dtype=mstype.float32)
    b = Tensor(B, dtype=mstype.float32)
    print('Initial parameters:', [a, b])

    learning_rate = 0.00488
    grad_all = ops.GradOperation(get_all=True)
    for i in range(250):

        output = grad_all(Net())(a, b, x, y)
        a = a + (-learning_rate * output[2][i])  # add_ 即 +=
        b = b + (-learning_rate * output[3][i])
        print(output[2][i])
        print(output[3][i])

    #
    x_data = x.asnumpy()
    plt.figure(figsize=(10, 7))
    xplot, = plt.plot(x_data, y.asnumpy(), 'o')
    yplot, = plt.plot(x_data, a.asnumpy() * x_data + b.asnumpy())
    plt.xlabel('X')
    plt.ylabel('Y')
    str1 = str(a.asnumpy()[0])+'X+' + str(b.asnumpy()[0])
    plt.legend([xplot, yplot], ['Data', str1])
    plt.show()
    #
    predictions = a * x_test + b
    print(predictions)
    print(y_test)
