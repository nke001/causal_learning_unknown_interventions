

## Learning Neural Causal Models from Unknown Interventions ##



This is a Pytorch implementation of the [Learning Neural Causal Models from Unknown Interventions](https://arxiv.org/abs/1910.01075) paper. Here we learn the causal model based on a meta-learning transfer objective from unknown intervention data. Please cite:

[Nan Rosemary Ke](https://nke001.github.io/)\*, [Olexa Bilaniuk](https://mila.quebec/en/person/olexa-bilaniuk/)\*, [Anirudh Goyal](https://anirudh9119.github.io/), [Stefan Bauer](https://www.is.mpg.de/~sbauer), [Hugo Larochelle](https://mila.quebec/en/person/hugo-larochelle/), [Chris Pal](https://mila.quebec/en/person/pal-christopher/), [Yoshua Bengio](https://mila.quebec/en/yoshua-bengio/)


    @article{ke2019learning,
        title={Learning Neural Causal Models from Unknown Interventions},
        author={Ke, Nan Rosemary and Bilaniuk, Olexa and Goyal, Anirudh and Bauer, Stefan and Larochelle, Hugo and Pal, Chris and Bengio, Yoshua},
        journal={arXiv preprint arXiv:1910.01075},
        year={2019}
    }


 

Meta-learning over a set of distributions can be interpreted as learning different types of parameters corresponding to short-term vs long-term aspects of the mechanisms underlying the generation of data. These are respectively captured by quickly-changing _parameters_ and slowly-changing _meta-parameters_. We present a new framework for meta-learning causal models where the relationship between each variable and its parents is modeled by a neural network, modulated by structural meta-parameters which capture the overall topology of a directed graphical model. Our approach avoids a discrete search over models in favour of a continuous optimization procedure. We study a setting where interventional distributions are induced as a result of a random intervention on a single unknown variable of an unknown ground truth causal model, and the observations arising after such an intervention constitute one meta-example. To disentangle the slow-changing aspects of each conditional from the fast-changing adaptations to each intervention, we parametrize the neural network into fast parameters and slow meta-parameters. We introduce a meta-learning objective that favours solutions _robust_ to frequent but sparse interventional distribution change, and which generalize well to previously unseen interventions. Optimizing this objective is shown experimentally to recover the structure of the causal graph. Finally, we find that when the learner is unaware of the intervention variable, it is able to infer that information, improving results further and focusing the parameter and meta-parameter updates where needed.


# Installation 

1. This code is based on Pytorch. The conda enviroment for running this code can be installed as follows,

```
conda env create -f environment.yml

```

2. Training code

```
# chain3
python run.py train --seed 1 --train_functional 10000 --mopt adam:5e-2,0.9 --gopt adam:5e-3,0.1 -v 500 --lsparse 0.1 --bs 256 --ldag 0.5 --predict 100 --temperature 1 -N 2 -p chain3

# fork3
python run.py train --seed 1 --train_functional 10000 --mopt adam:5e-2,0.9 --gopt adam:5e-3,0.1 -v 500 --lsparse 0.1 --bs 256 --ldag 0.5 --predict 100 --temperature 1 -N 2 -p fork3

# collider3
python run.py train --seed 1 --train_functional 10000 --mopt adam:5e-2,0.9 --gopt adam:5e-3,0.1 -v 500 --lsparse 0.1 --bs 256 --ldag 0.5 --predict 100 --temperature 1 -N 2 -p collider3

# confounder3
python run.py train --seed 1 --train_functional 10000 --mopt adam:5e-2,0.9 --gopt adam:5e-3,0.1 -v 500 --lsparse 0.1 --bs 256 --ldag 0.5 --predict 100 --temperature 1 -N 2 -p confounder3
```


`--seed` specifies the random seed

`--mopt` specifies the optimizer and learning rate used to train the functional parameters

`--gopt` specifies the optimizer and learning rate used to train the structural parameters

`--predict` specifies the number of samples used for predicting the intervened node. Specifying 0 for this argument uses the groundtruth intervention node.

`--temperature` specifies the temperature setting fot the softmax for the groundtruth structured causal model.

`-N` specifies the number of categories for the categorical distribution

`-M` specifies the number of discrete random variables

`--graph` allows one to specify via the command-line several causal DAG skeletons.

`-p` specifies, by name, one of several `--graph` presets for groundtruth causal graphs (e.g. `chain3`).

`--train_functional` specifies how many iterations to train the functional parameters.
