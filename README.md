# Project: Optimization for Machine Learning and Neural Networks
### Momentum and Nesterov's Accelerated Gradiend Methods
Momentum method is a technique for accelerating the stochastic gradient descent(SGD) method, the momentum update can be motivated from a physical perspective by introducing a velocity component. A velocity vector is accumulated in directions that consistenly reduce the cost function, allowing update the parameters in a direction that is more effective than steepest descent. Nesterov's accelerated gradient(NAG) is a variant of momentum method, the main difference is that we also apply a momentum update to the parameters and the gradient is computed with these updated parameters in each step. Momentum methods approach most of the time achieve better convergence rates on deep neural networks compared with vanilla SGD.

#### Author: Minh Nhat Do - Franz Franco Gallo



## Step to run project

#### Step 1: Clone the repository to local machine

#### Step 2: Go the the repository diretory

#### Step 3:  Run visualization with Toy data example 1

To reproduce the results run the Toy_example_regression.py script as follows: 

```shellscript
python Toy_example_regression.py

```

#### Step 4:  Run visualization with Toy data example 2

```shellscript
python -m toy_example --epochs=100 --gamma=0.9 --eta=1
```
  **Parameters**
  -  ```epochs``` is parameters of Number of Epochs
  -  ```gamma``` is parameters of momentum values input (gammma) (from 0 - 1)
  -  ```eta``` is parameters of Learning rate (Theta)

  Plots includes:
  - **1** graph on parameters and error on Momentum method
  - **1** graph on parameters and error on NAG method
  - **1** animation graph on visualizing how it changed on each epochs



#### Step 5:  Run visualization with Neuron network example


```shellscript
python -m nn_example --epochs=40 --gamma=0.9 --eta=1
```
  **Parameters**
  -  ```epochs``` is parameters of Number of Epochs
  -  ```gamma``` is parameters of momentum values input (gammma) (from 0 - 1)
  -  ```eta``` is parameters of Learning rate (Theta)

  Plots includes:
  - **1** graph on example generated data 
  - **1** graph on comparing Loss on 3 method (SGD with out Momentum, Momentumm, and NAG)
  - **1** graph on comparing Accuracy on 3 method (SGD with out Momentum, Momentumm, and NAG)



## Result

### Toy data example approach 1

**Result on Mini batch gradient descent**
<img src="https://github.com/frafaga229/Project_Opt4ML/blob/main/results/Mini%20batch%20Gradient%20Descent.png" width="800"><br>
**Result on Momentum method**
<img src="https://github.com/frafaga229/Project_Opt4ML/blob/main/results/Momentum%20method.png" width="800"><br>
**Result on Nesterov's Accelerated Gradient**

<img src="https://github.com/frafaga229/Project_Opt4ML/blob/main/results/Nesterov's%20Accelerated%20Gradient.png" width="800"><br>

### Toy data example approach 2
**Visualization on learning sigmoid parameters process  of Momentum method and Nesterov's Accelerated Gradient**
<img src="https://github.com/frafaga229/Project_Opt4ML/blob/main/img/toy_example.png" width="700"><br>

### Neuron Network example
 **Loss values on 3 methods SGD, Momentum, Nesterov's Accelerated Gradient**
<img src="https://github.com/frafaga229/Project_Opt4ML/blob/main/img/nn_loss.png" width="700"><br>
**Accuracy values on 3 methods SGD, Momentum, Nesterov's Accelerated Gradient**
<img src="https://github.com/frafaga229/Project_Opt4ML/blob/main/img/nn_acc.png" width="700"><br>
