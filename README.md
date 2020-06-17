

## Learning Neural Causal Models from Unknown Interventions ##



This is a Pytorch implementation of the [Learning Neural Causal Models from Unknown Interventions](https://arxiv.org/abs/1910.01075) paper. Here we learn the causal model based on a meta-learning transfer objective from unknown intervention data. Please cite:

[Nan Rosemary Ke](https://nke001.github.io/)\*, [Olexa Bilaniuk](https://mila.quebec/en/person/olexa-bilaniuk/)\*, [Anirudh Goyal](https://anirudh9119.github.io/), [Stefan Bauer](https://www.is.mpg.de/~sbauer), [Hugo Larochelle](https://mila.quebec/en/person/hugo-larochelle/), [Chris Pal](https://mila.quebec/en/person/pal-christopher/), [Yoshua Bengio](https://mila.quebec/en/yoshua-bengio/)


    @article{ke2019learning,
        title={Learning Neural Causal Models from Unknown Interventions},
        author={Ke, Nan Rosemary and Bilaniuk, Olexa and Goyal, Anirudh and Bauer, Stefan and Larochelle, Hugo and Pal, Chris and Bengio, Yoshua},
        journal={arXiv preprint arXiv:1910.01075},
        year={2019}
    }




# Installation 

1. This code is based on Pytorch. The conda enviroment for running this code can be installed as follows,

```
conda env create -f environment.yml

pip install -e .

```
CPU features required: AVX2, FMA (Intel Haswell+) 

---
2. Training code

```
# chain3
python run.py train --seed 1 --train_functional 10000 --mopt adam:5e-2,0.9 --gopt adam:5e-3,0.1 -v 500 --lsparse 0.1 --bs 256 --ldag 0.5 --predict 100 --temperature 1 --limit-samples 500 -N 2 -p chain3  

# fork3
python run.py train --seed 1 --train_functional 10000 --mopt adam:5e-2,0.9 --gopt adam:5e-3,0.1 -v 500 --lsparse 0.1 --bs 256 --ldag 0.5 --predict 100 --temperature 1 --limit-samples 500 -N 2 -p fork3

# collider3
python run.py train --seed 1 --train_functional 10000 --mopt adam:5e-2,0.9 --gopt adam:5e-3,0.1 -v 500 --lsparse 0.1 --bs 256 --ldag 0.5 --predict 100 --temperature 1 --limit-samples 500 -N 2 -p collider3

# confounder3
python run.py train --seed 1 --train_functional 10000 --mopt adam:5e-2,0.9 --gopt adam:5e-3,0.1 -v 500 --lsparse 0.1 --bs 256 --ldag 0.5 --predict 100 --temperature 1 --limit-samples 500 -N 2 -p confounder3
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

`--limit-samples` specifies the number of samples used per intervention. Suggest to use 500 for graphs of size < 10 and 1000 for graphs size between 10 and 15.


By default, the models and log files are stored in the `work` directory.
 



---
3. Extracting useful information from log files.




Strip all the SLURM output of interruptions:

```
scripts/strip_interrupts.py slurm-%j.out         > STRIPPED_LOGFILE.txt
```

Extract gamma CE timeseries

```
scripts/series_gammace.py   STRIPPED_LOGFILE.txt > gammace.txt
```

Compute AUROC-over-time

```
scripts/series_auc.py       STRIPPED_LOGFILE_1.txt STRIPPED_LOGFILE_2.txt STRIPPED_LOGFILE_3.txt STRIPPED_LOGFILE_4.txt STRIPPED_LOGFILE_5.txt > auc.txt

```
