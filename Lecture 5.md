- transfer learning with CNNs for pretraining and fine tuning on a smaller dataset.
- ![[Pasted image 20250424152038.png]]
- 1960: mark 1 perceptron, used a binary step function instead of an activation function
- 1986: backpropagation
- 2006: hinton and salakhutdinov: deep NN

## Activation Functions
1. Sigmoid Function
	- squashes numbers to range [0,1]
	- problem is that sigmoid are not zero centered.
	- the gradients are always all +ve or all -ve.
	- exp() is more compute expensive compared to other activation functions
2. Tanh
	- squashes numbers to [-1.1]
	- zero centered
	- still kills gradients when saturated
3. ReLU
	- f(x) = max(0,x)
	- does not saturate in +ve region
	- very computationally efficient
	- converges faster
	- not zero centered still
	- still kills gradients(if not activated once ,will never activate)

4. Leaky ReLU
	- f(x) = max(0.01x, x)
	- does not saturate
	- does not die

5. Maxout "Neuron"
	- generalizes relu and leaky relu
	- does not have nonlinearity
	- $$\max(w_1^Tx+b_1,w_2^Tx+b_2)$$
- PCA : dimensionality reduction
	-  removes the arbitrary collinear features
- to get rid of saturation , there should be a reasonable initialization 

- ### Batch Normalization
	- $$\widehat{x}^{(k)}=\frac{x^{(k)}-\mathrm{E}[x^{(k)}]}{\sqrt{\mathrm{Var}[x^{(k)}]}}$$
	- mini batch -> unit gaussian activations for every feature
	- Why does this happen?
		-  what stops gradient flow?
	- improves gradient flow through the network
	- allows for higher learning rates
	- reduces the strong dependence on initialization
	- the mean/std are not computed based on the batch 