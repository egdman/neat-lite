This is a standalone version of the Neural Evolution through Augmenting Topologies (NEAT) [Stanley2002](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf) learning algorithm that I used for gait learning in modular robots in the [tol-revolve project](https://github.com/egdman/tol-revolve/tree/tol-env). I am generalizing the algorithm and making it into its own python package.

Currently the algorithm uses a simplified fitness sharing process that does not maintain species.

**Installation**

`pip install -e git+https://github.com/egdman/neat-lite.git@master#egg=neat`


The `examples/` directory contains a simple implementation of the neural network and an example of learning a XOR network. The neural network implementation is not really a part of this package, it's only included as an example.

The XOR example has 2 phases: augmentation (developing solutions through augmenting their topologies) and reduction (trying to simplify topologies of the solutions while keeping their fitness values high). 