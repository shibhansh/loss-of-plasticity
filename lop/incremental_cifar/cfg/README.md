# Structure of Config Files

Below is the config file used to run continual backpropagation.

```json
  "_model_description_": "Continual backpropagation algorithm",

  "_comment_1_": "Data paths; change data_path and results_dir to your convenience",
  "data_path": "",
  "results_dir": "",
  "experiment_name":  "continual_backpropagation",
  "num_workers":  12,

  "_comment_2_": "Learner Parameters",
  "stepsize":  0.1,
  "weight_decay": 0.0005,
  "momentum": 0.9,
  
  "_comment_3_": "Shrink and perturb parameters",
  "noise_std": 0.0,

  "_comment_4_": "Continual Backprop Parameters",
  "use_cbp": true,
  "replacement_rate": 0.00001,
  "utility_function": "contribution",
  "maturity_threshold": 1000,

  "_comment_5_": "Network resetting parameters",
  "reset_head": false,
  "reset_network": false,
  "early_stopping": true
```

Here's a detail description of each entry except for entries enclosed by underscores since they're 
not used for running the experiment:

* `data_path`: path to the directory containing the CIFAR-100 data. 
    When not provided or is set to an empty string, the default is `./loss-of-plasticity/lop/incremental_cifar/data/`
* `results_dir`: path to directory in which to store the experiment results
    When not provided or is set to an empty string, the default is `./loss-of-plasticity/lop/incremental_cifar/results/`
* `experiment_name`: name of your choosing. 
    If it's not unique, the script will override any information found at `$results_dir/$experiment_name`
* `num_workers`: integer corresponding to the number of workers used for the torch data loader. 
    Don't use a number greater than the number of available cpu cores.
* `stepsize`: float corresponding to the initial stepsize or learning rate of SGD.
    The learning rate schedule will still be applied to this stepsize as described in the paper.
* `weight_decay`: float; L2 penalty term.
* `noise_std`: float; standard deviation of the Gaussian Noise for the shrink and perturb algorithm.
* `use_cbp`: bool; whether to use continual backpropagation.
    If false, you may omit the following three parameters.
* `replacement_rate`: float; continual backpropagation parameter.
* `utility_function`: string; continual backpropagation parameter.
    There are only two accepted values, weight or contribution.
* `maturity_threshold`: positive integer; continual backprop parameter.
* `reset_head`: bool; whether to reinitialize the output layer of the network at the start of each new task.
* `reset_network`: bool; whether to reinitialize the network at the start of each new task.
* `early_stopping`: bool; whether to use early stopping.
    Note that all the algorithms in the main paper use early stopping. 
    We noticed that not using early stopping also shows loss of plasticity, but it also significantly
    deteriorates performance due to over-fitting. 



