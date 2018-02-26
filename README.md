# HyperGradientDescent

This repo contains TensorFlow implementation of the Adam-HD algorithm discribed in the paper [**Online Learning Rate Adaptation with Hypergradient Descent**](https://arxiv.org/abs/1703.04782). 

In addition, we reproduced some results presented in the paper.

Here you see the results for training CNN on CIFAR-10 with batch size 128 near 250 epoch using ADAM algorithm with learning rates $\alpha = [10^{-3},10^{-4},10^{-5}]$.
[ADAM](./ADAM.png)
Here are the results for ADAM-HD algorithm with different iniutial learning rates $\alpha_0 = [10^{-3},10^{-4},10^{-5}]$
The hyperlearning rate $\beta$ is fixed and equal to $10^{-7}$ as was suggested in the paper. 
[ADAM-HD](./ADAM-HD.png)

We plot the evolution of learning rates during training. As wqas described in the paper. the learning rate increases during first stage of the training and then goes down to the $10^{-5}$.  
[Learning rate adaptation](./alpha.png)



