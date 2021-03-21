### Project: Optimization for Machine Learning and Neural Networks
#### Momentum and Nesterov's Accelerated Gradiend Methods
Momentum method is a technique for accelerating the stochastic gradient descent(SGD) method, the momentum update can be motivated from a physical perspective by introducing a velocity component. A velocity vector is accumulated in directions that consistenly reduce the cost function, allowing update the parameters in a direction that is more effective than steepest descent. Nesterov's accelerated gradient(NAG) is a variant of momentum method, the main difference is that we also apply a momentum update to the parameters and the gradient is computed with these updated parameters in each step. Momentum methods approach most of the time achieve better convergence rates on deep neural networks compared with vanilla SGD.

#### Toy examples
##### Regression
To reproduce the results run the Toy_example_regression.py script as follows: python Toy_example_regression.py


##### Minh Nhat toy example

#### Neural Networks
