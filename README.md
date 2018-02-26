# HyperGradientDescent

This repo contains TensorFlow implementation of the Adam-HD algorithm discribed in the paper [**Online Learning Rate Adaptation with Hypergradient Descent**](https://arxiv.org/abs/1703.04782). 

In addition, we reproduced some results presented in the paper.

Here you see the results for training CNN on CIFAR-10 with batch size 128 near 100 epoch using ADAM algorithm with learning rates ![img](http://latex.codecogs.com/svg.latex?%24%5Calpha%3D%5B10%5E%7B-3%7D%2C10%5E%7B-4%7D%2C10%5E%7B-5%7D%5D%24%0D%0A).

![ADAM](./ADAM.png)

Here are the results for ADAM-HD algorithm with different iniutial learning rates  ![img](http://latex.codecogs.com/svg.latex?%24%5Calpha_0%3D%5B10%5E%7B-3%7D%2C10%5E%7B-4%7D%2C10%5E%7B-5%7D%5D%24%0D%0A)
The hyper learning rate ![img](http://latex.codecogs.com/svg.latex?%5Cbeta%3D10%5E%7B-7%7D) is fixed. The value is the same as was in the poaper.
![ADAM-HD](./ADAM-HD.png)

We plot the evolution of learning rates during training. As wqas described in the paper. the learning rate increases during first stage of the training and then goes down to the ![img](http://latex.codecogs.com/svg.latex?10%5E%7B-5%7D).  
![Learning rate adaptation](./alpha.png)



