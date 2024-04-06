import numpy as np

def count_parameters(model):
    """统计模型参数"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Scaler:
    def __init__(self, data, missing_value=np.inf):
        values = data[data != missing_value]
        self.mean = values.mean()
        self.std = values.std()

    def transform(self, data):  #数据标准化
        return (data - self.mean) / (self.std + 1e-8)

    def inverse_transform(self, data):  #还原数据
        return data * self.std + self.mean


def add_indent(str_, num_spaces):  #在每行字符串前面添加缩进
    s = str_.split('\n')
    s = [(num_spaces * ' ') + line for line in s]
    return '\n'.join(s)


def num_parameters(layer):   #统计某层的参数量
    def prod(arr):
        cnt = 1
        for i in arr:
            cnt = cnt * i
        return cnt

    cnt = 0
    for p in layer.parameters():
        cnt += prod(p.size())
    return cnt
