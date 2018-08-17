# python-neuroevolution
An implementation of "Conventional Neuroevolution" - evolving ANNs without backpropegation

An implementation of a simple 3 layer ANN, which evolvs it's weights using a Genetic Algorithim. 
Implemented with Numpy

My main motivation for avoiding backpropegation is that I don't intuitivly understand the math behind it. I understand the mechanism that a genetic algorithim uses to evolve. Luckily, this idea has been heavily studied and is called "Neroevolution"

Conventional Neuroevolution is considered to be inferior to SGD - but Neurevolution of the network topology has been reported to be competative with traditional techniques. My next steps for this code include implementing a GA to vary things like network size