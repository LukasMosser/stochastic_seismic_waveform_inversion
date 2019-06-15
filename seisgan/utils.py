import torch
import random
import numpy as np
import os
import errno


def tn(n):
    return n.data.cpu().numpy()


def output_to_tensorboard(writer, loss_vars, loss_names, iteration):
    for ls, name in zip(loss_vars, loss_names):
        writer.add_scalar(name, ls, global_step=iteration)


def output_losses(logger, loss_vars, loss_names, run_number, iteration):
    for ls, name in zip(loss_vars, loss_names):
        logger.info('Model Inference Run: {:1d}, Iteration: {:1d}, {}: {:1.2f}'.format(run_number, iteration, name, float(ls)))


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


def make_dir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise