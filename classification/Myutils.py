import random
import os
import torch.backends.cudnn as cudnn
import torch


def init_random_seed(manual_seed):
    """Init random seed."""
    seed = True
    if manual_seed is None:
        seed = random.randint(1, 10000)
    else:
        seed = manual_seed
    print("use random seed: {}".format(seed))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
def save_model(net, filename):
    """Save trained model."""
    if not os.path.exists("abc"):
        os.makedirs("abc")
    torch.save(net.state_dict(),
               os.path.join("abc", filename))
    print("save pretrained model to: {}".format(os.path.join("abc",
                                                             filename)))
def init_model(net, restore):
    """Init models with cuda and weights."""
    # init weights of model
    # net.apply(init_weights)      #apply函数用于初始化
    # print("网络的初始化函数")
    # restore model weights
    if restore is not None and os.path.exists(restore):
        print("给restored赋值为true")
        net.load_state_dict(torch.load(restore))
        net.restored = True
        print("Restore model from: {}".format(os.path.abspath(restore)))
    # check if cuda is available
    if torch.cuda.is_available():
        cudnn.benchmark = True
        net.cuda()
    return net