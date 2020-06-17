import nauka
import numpy as np
import os, sys, time, pdb
import torch

from   causal.models import CategoricalWorld



class Experiment:
    def __init__(self, a):
        self.a = type(a)(**a.__dict__)
        self.a.__dict__.pop("__argp__", None)
        self.a.__dict__.pop("__argv__", None)
        self.a.__dict__.pop("__cls__",  None)
    
    def run(self):
        ### Create world from graph file
        world = CategoricalWorld(self.a)
        
        ### Node Arity
        arity = torch.as_tensor(world.N, dtype=torch.int8).unsqueeze(0)
        
        ### Sampling
        samples = torch.empty((world.M+1,
                               self.a.num_interventions,
                               self.a.num_samples,
                               world.M), dtype=torch.int32)
        #     Interventional Data Generation
        for i in range(world.M):
            for j in range(self.a.num_interventions):
                with world.intervention(i):
                    samples[i,j] = next(world.sampleiter(self.a.num_samples)).transpose(0,1)
        #     Observational Data Generation
        for j in range(self.a.num_interventions):
            samples[world.M,j] = next(world.sampleiter(self.a.num_samples)).transpose(0,1)
        #     Convert from one-hot to 1-based index for MATLAB
        samples = samples.add_(1).view(-1, world.M)
        
        ### Clamped Mask
        clamped = torch.zeros_like(samples, dtype=torch.uint8)
        for i in range(world.M):
            clamped[(i+0)*self.a.num_samples*self.a.num_interventions:
                    (i+1)*self.a.num_samples*self.a.num_interventions, i] = 1
        clamped = torch.cat([torch.zeros_like(samples), clamped], 1).byte()
        
        ### Padded DAG
        #     For BDAGL, the matrix is (mostly) upper triangular, and every
        #     node has a shadow node that represents its intervened self,
        #     such as it were.
        dag  = world.gammagt
        zero = torch.zeros_like(dag)
        dag  = torch.cat([torch.cat([dag,  zero], 1),
                          torch.cat([zero, zero], 1),], 0)
        dag.diagonal(+world.M).fill_(1)
        dag  = dag.byte()
        
        #
        # Print
        #
        os.makedirs(self.a.dumpDir, exist_ok=True)
        np.savetxt(os.path.join(self.a.dumpDir, "nodeArities.csv"),
                   arity.numpy(),   fmt="%d", delimiter=",")
        np.savetxt(os.path.join(self.a.dumpDir, "dag.csv"),
                   dag.numpy(),     fmt="%d", delimiter=",")
        np.savetxt(os.path.join(self.a.dumpDir, "samples.csv"),
                   samples.numpy(), fmt="%d", delimiter=",")
        np.savetxt(os.path.join(self.a.dumpDir, "clampedMask.csv"),
                   clamped.numpy(), fmt="%d", delimiter=",")

