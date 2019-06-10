import torch
import random
import numpy as np

def printgradnorm(self, grad_input, grad_output):
    print('Inside ' + self.__class__.__name__ + ' backward')
    print('Inside class:' + self.__class__.__name__)
    print('')
    print('grad_input: ', type(grad_input))
    print('grad_input[0]: ', type(grad_input[0]))
    print('grad_output: ', type(grad_output))
    print('grad_output[0]: ', type(grad_output[0]))
    print('')
    print('grad_input size:', grad_input[0].size())
    print('grad_output size:', grad_output[0].size())
    print('grad_input norm:', grad_input[0].data.norm())


def create_grad_storage(x_in, x_out):
    def grads(self, grad_input, grad_output):
        print('Inside ' + self.__class__.__name__ + ' backward')
        print('Inside class:' + self.__class__.__name__)
        print('')
        print('grad_input: ', type(grad_input))
        print('grad_input[0]: ', type(grad_input[0]))
        print('grad_output: ', type(grad_output))
        print('grad_output[0]: ', type(grad_output[0]))
        print('')
        print('grad_input size:', grad_input[0].size())
        print('grad_output size:', grad_output[0].size())
        print('grad_input norm:', grad_input[0].data.norm())
        x_in = grad_input
        x_out = grad_output
    return grads


def set_seed(seed):
    """
    Set the random number generator and turn off the cudnn benchmarks and backends to make truly deterministic
    For reproducibility purposes
    :param seed: random number generator seed
    :return: True on Success
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    return True