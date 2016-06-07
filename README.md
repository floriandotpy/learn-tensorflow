# learn-tensorflow

My own examples in getting to know tensorflow.

## 1. CIFAR-10

Still ugly code, unneeded debug output and to little training samples and iterations. 
Mainly tried to get this running at all with a tiny CNN.

- Download [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) by hand: Grab _CIFAR-10 python version_, extract and place it in `cifar-10/data`
- `cd cifar-10`
- `python train.py`. Grab a coffee.
- To view Tensorboard, run `tensorboard --logdir=cifar-10/logs`