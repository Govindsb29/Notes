## Backpropagation
**Computational Graph** #computational-graphs
$$\boxed{f=Wx}  $$
$$\boxed{L_{i}=\sum_{j\neq y_{i}}\max(0,s_{j}-s_{y_{i}}+1)}
$$
- alexnet: convolutional networks, neural turing machine(deepmind, differentiable completely ), these have huge #computational-graphs .
-  intermediate terms in gradient descent:
$$ f=qz\quad\frac{\partial f}{\partial q}=z,\frac{\partial f}{\partial z}=q$$
- $$ f(w,x)=\frac{1}{1+e^{-(w_0x_0+w_1x_1+w_2)}}\quad\boxed{\sigma(x)=\frac{1}{1+e^{-x}}}$$
- $$ \quad\text{sigmoid function}  
\frac{d\sigma(x)}{dx}=\frac{e^{-x}}{\left(1+e^{-x}\right)^{2}}=\left(\frac{1+e^{-x}-1}{1+e^{-x}}\right)\left(\frac{1}{1+e^{-x}}\right)=\left(1-\sigma(x)\right)\sigma(x)$$

- max gate: no change in gradient
- add gate: no cahnge
- mul gate: switch the values as gradients
- ![[Pasted image 20250424140032.png]]
- the matrix if the input and output vectors is known as #jacobian_matrix
- #exercise SVM/Softmax forward/backward computation
- implementation of 2 layer neural network is simple for binary classification that uses syn or synapses .

- history of neural networks from biological neural networks
![[Pasted image 20250424141446.png]]
- more neurons = more capacity