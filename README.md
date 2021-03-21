# Project_Opt4ML



## Step to run project

#### Step 1: Clone the repository to local machine

#### Step 2: Go the the repository diretory

#### Step 3:  Run visualization with Toy data example

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

#### Step 4:  Run visualization with Neuron network example

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
