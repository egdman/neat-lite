This is a standalone version of the Neural Evolution through Augmenting Topologies (NEAT) learning algorithm that I used for gait learning in modular robots in the [tol-revolve project](https://github.com/egdman/tol-revolve/tree/tol-env). I am generalizing the algorithm and making it into its own python package.

**Installation**

`pip install -e git+https://github.com/egdman/neat-lite.git@master#egg=neat`


The `examples/` directory contains a simple implementation of the neural network and an example of learning a XOR network. The neural network implementation is not really a part of this package - implementing the network for the client's purpose is client's job.

The XOR example has 2 phases: augmentation (developing solutions through augmenting their topologies) and reduction (trying to simplify topologies of the solutions while keeping their fitness values high). 