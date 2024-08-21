# Implementation of Continual Backpropagation
This directory contains different implementations of continual backpropagation. The results in the paper for feed-forward, convolutional, and residual networks in the paper are generated using `cbp.py,` 
`convCBP.py,` `res_gnt.py` respectively. 

`cbp_linear.py` and `cbp_conv.py` contain a newer and easier-to-use implementation of continual backpropagation.
This implementation allows you to use continual backpropagation like a layer in a network (similar to dropout or batch norm).

To use CBP as a layer, define a CBP layer in the network and make sure that activation passes through the CBP layer during the forward pass. See [../nets/conv_net2.py](../nets/conv_net2.py) for an example.

CBPLinear takes the following arguments. The first four are important to fully describe the algorithm. The remaining are important for functioning of the algorithm but default values work well:
* `in_layer`: The layer containing the incoming weights of hidden units. 
* `out_layer`: The layer containing the outgoing weights of hidden units. 
* `replacement_rate`: An important hyperparameter of continual backpropagation. It is the number of units replaced per step. 
Values between 1e-4 to 1e-6 generally perform well.
* `maturity_threshold`: An important hyperparameter of continual backpropagation. It is the number of steps for which a unit is protected from replacement.
Values between 100-10,000 generally perform well.
* `decay_rate`: A hyperparameter of continual backpropagation that controls the quality of the utility estimate. 
The default is `0.99`; it seems to work well in all cases. A value of `0` can be used to speed up the algorithm at a minimal reduction in performance.
* `init`: Name of the distribution used to initialize the weights of the network. The default is `kaiming`.
* `act_type`: Name of the non-linear activation of the hidden units. The default is `relu`.
* `util_type`: Name of the utility measure. The default is `contribution`. It works well with Relu-type activations.
* `ln_layer`: Optional layer norm layer before or after the activation.
* `bn_layer`: Optional batch norm layer before or after the activation.

Note that the newer  implementations in `cbp_linear.py` and `cbp_conv.py` are not as thoroughly tested, 
so there might be small bugs in this implementation.