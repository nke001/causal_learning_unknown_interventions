# -*- coding: utf-8 -*-
import contextlib
import itertools
import os, sys, time, pdb
import numpy                                as np
import torch
import torch.nn


from   torch.distributions              import (OneHotCategorical)
from   torch.nn                         import (Module,Parameter,)



class World(Module):
    def parameters(self):
        s = set(self.structural_parameters())
        l = [p for p in super().parameters() if p not in s]
        return iter(l)
    
    def structural_parameters(self):
        return iter([self.gamma])



class CategoricalWorld(World):
    def __init__(self, a, *args, **kwargs):
        super().__init__()
        self.a = a
        self.init(*args, **kwargs)
    
    def init(self, *args, **kwargs):
        self.initgraph()
        
        self.register_buffer   ("W0gt",    torch.FloatTensor(self.M, self.Hgt, self.M, self.N))
        self.register_buffer   ("B0gt",    torch.FloatTensor(self.M, self.Hgt                ))
        self.register_buffer   ("W1gt",    torch.FloatTensor(self.M, self.N, self.Hgt        ))
        self.register_buffer   ("B1gt",    torch.FloatTensor(self.M, self.N                  ))
        
        self.register_parameter("gamma",   Parameter(torch.zeros_like(self.gammagt)))
        
        self.register_parameter("W0slow",  Parameter(torch.zeros((self.M, self.Hlr, self.M, self.N), dtype=self.W0gt.dtype)))
        self.register_parameter("B0slow",  Parameter(torch.zeros((self.M, self.Hlr                ), dtype=self.B0gt.dtype)))
        self.register_parameter("W1slow",  Parameter(torch.zeros((self.M, self.N, self.Hlr        ), dtype=self.W1gt.dtype)))
        self.register_parameter("B1slow",  Parameter(torch.zeros((self.M, self.N                  ), dtype=self.B1gt.dtype)))
        
        self.register_parameter("W0fast",  Parameter(torch.zeros_like(self.W0slow)))
        self.register_parameter("B0fast",  Parameter(torch.zeros_like(self.B0slow)))
        self.register_parameter("W1fast",  Parameter(torch.zeros_like(self.W1slow)))
        self.register_parameter("B1fast",  Parameter(torch.zeros_like(self.B1slow)))
        
        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=0.1)
        
        for i in range(self.M):
            torch.nn.init.orthogonal_(self.W0slow[i])
            torch.nn.init.orthogonal_(self.W1slow[i])
        torch.nn.init.uniform_(self.B0slow,    -.1, +.1)
        torch.nn.init.uniform_(self.B1slow,    -.1, +.1)
        torch.nn.init.uniform_(self.gamma, -.1, +.1)
        with torch.no_grad(): self.gamma.diagonal().fill_(float("-inf"))
        
        self.alterdists()
        self.reconstrain_theta(force=self.a.structural_init)
    
    @property
    def Hgt(self):
        if self.a.hidden_truth is None:
            if self.M > self.N: return 4*self.M
            else:               return 4*self.N
        else:
            return int(self.a.hidden_truth)
    
    @property
    def Hlr(self):
        if self.a.hidden_learn is None:
            if self.M > self.N: return 4*self.M
            else:               return 4*self.N
        else:
            return int(self.a.hidden_learn)
    
    def initgraph(self):
        if self.a.graph is None:
            self.M = self.a.num_vars
            self.N = self.a.num_cats
            self.initgammagt()
            
            expParents = self.a.num_parents
            idx        = np.arange(self.M).astype(np.float32)[:,np.newaxis]
            idx_maxed  = np.minimum(idx*0.5, expParents)
            p          = np.broadcast_to(idx_maxed/(idx+1), (self.M, self.M))
            B          = np.random.binomial(1, p)
            B          = np.tril(B, -1)
            self.gammagt.copy_(torch.as_tensor(B))
        else:
            self.M = self.a.num_vars
            self.N = self.a.num_cats
            self.initgammagt()
            
            for g in self.a.graph:
                for e in g.split(","):
                    if e == "": continue
                    nodes = e.split("->")
                    if len(nodes) <= 1: continue
                    nodes = [int(n) for n in nodes]
                    for src, dst in zip(nodes[:-1], nodes[1:]):
                        if dst > src:
                            self.gammagt[dst,src] = 1
                        elif dst == src:
                            raise ValueError("Edges are not allowed from " +
                                             str(src) + " to oneself!")
                        else:
                            raise ValueError("Edges are not allowed from " +
                                             str(src) + " to ancestor " +
                                             str(dst) + " !")
        
        return self
    
    def initgammagt(self):
        if not hasattr(self, "gammagt"):
            self.register_buffer("gammagt", torch.empty((self.M, self.M)))
        self.gammagt.zero_()
    
    def vizualize_gamma(self):
        """Generate a rendering of a gamma matrix vs its ground truth."""
        
        with torch.no_grad():
            RGBPALETTE = torch.as_tensor([
                [213, 62, 79], # Ruby Red
                [244,109, 67],
                [253,174, 97],
                [254,224,139],
                [255,255,191], # Pale Yellow
                [230,245,152],
                [171,221,164],
                [102,194,165],
                [ 50,136,189], # Deep Blue
            ][::-1], dtype=torch.float32)
            
            GAMMAGT  = self.gammagt
            GAMMALR  = self.gamma.sigmoid()
            INDEXGT  = GAMMAGT.float().mul(len(RGBPALETTE)-1)
            INDEXLR  = GAMMALR.float().mul(len(RGBPALETTE)-1)
            INDEXGTL = INDEXGT.floor().long()
            INDEXGTF = INDEXGT.float()-INDEXGTL.float()
            INDEXGTU = INDEXGT.ceil ().long()
            INDEXLRL = INDEXLR.floor().long()
            INDEXLRF = INDEXLR.float()-INDEXLRL.float()
            INDEXLRU = INDEXLR.ceil ().long()
            PIXELGTL = torch.index_select(RGBPALETTE, 0, INDEXGTL.view(-1))
            PIXELGTU = torch.index_select(RGBPALETTE, 0, INDEXGTU.view(-1))
            PIXELLRL = torch.index_select(RGBPALETTE, 0, INDEXLRL.view(-1))
            PIXELLRU = torch.index_select(RGBPALETTE, 0, INDEXLRU.view(-1))
            PIXELGTL = PIXELGTL.view(INDEXGTL.shape + RGBPALETTE.shape[1:])
            PIXELGTU = PIXELGTU.view(INDEXGTU.shape + RGBPALETTE.shape[1:])
            PIXELLRL = PIXELLRL.view(INDEXLRL.shape + RGBPALETTE.shape[1:])
            PIXELLRU = PIXELLRU.view(INDEXLRU.shape + RGBPALETTE.shape[1:])
            PIXELGT  = PIXELGTU*INDEXGTF.unsqueeze(-1) + PIXELGTL*(1-INDEXGTF).unsqueeze(-1)
            PIXELLR  = PIXELLRU*INDEXLRF.unsqueeze(-1) + PIXELLRL*(1-INDEXLRF).unsqueeze(-1)
            PIXELGT  = PIXELGT.round().clamp(0., 255.).byte()
            PIXELLR  = PIXELLR.round().clamp(0., 255.).byte()
            PIXELLR  = PIXELLR.repeat_interleave(20, dim=0) \
                              .repeat_interleave(20, dim=1)
            for x,y in [ ( 7, 8), ( 7, 9), ( 7,10), ( 7,11),
                ( 8, 7), ( 8, 8), ( 8, 9), ( 8,10), ( 8,11), ( 8,12),
                ( 9, 7), ( 9, 8), ( 9, 9), ( 9,10), ( 9,11), ( 9,12),
                (10, 7), (10, 8), (10, 9), (10,10), (10,11), (10,12),
                (11, 7), (11, 8), (11, 9), (11,10), (11,11), (11,12),
                         (12, 8), (12, 9), (12,10), (12,11)]:
                PIXELLR[x::20,y::20] = PIXELGT
            
            return PIXELLR
    
    def alterdists(self):
        """For randomly-initialized distributions, alter them entirely."""
        for i in range(self.M):
            torch.nn.init.orthogonal_(self.W0gt[i], 2.5)
            torch.nn.init.orthogonal_(self.W1gt[i], 2.5)
            torch.nn.init.uniform_(self.B0gt[i], -1.1, +1.1)
            torch.nn.init.uniform_(self.B1gt[i], -1.1, +1.1)
        return self
    
    @contextlib.contextmanager
    def intervene(self, *args, **kwargs):
        """Perform an intervention, then undo it."""
        
        class CategoricalWorldIntervention:
            def __init__(salf, i=None):
                salf.node = [np.random.randint(0, self.M) if i is None else i]
            
            def __iter__(salf):
                return iter(salf.node)
            
            def do(salf):
                with torch.no_grad():
                    salf.clones = [t.clone() for t in self.parameters_gt()]
                for i in salf.node:
                    torch.nn.init.orthogonal_(self.W0gt[i], 2.5)
                    torch.nn.init.orthogonal_(self.W1gt[i], 2.5)
                    torch.nn.init.uniform_(self.B0gt[i], -1.1, +1.1)
                    torch.nn.init.uniform_(self.B1gt[i], -1.1, +1.1)
            
            def undo(salf):
                with torch.no_grad():
                    for tnew, told in zip(self.parameters_gt(), salf.clones):
                        tnew.copy_(told)
        
        intervention = CategoricalWorldIntervention(*args, **kwargs)
        try:
            intervention.do()
            yield intervention
        except:  raise
        finally: intervention.undo()
    
    def configpretrainiter(self, full_connect=False):
        """
        Sample a configuration for pretraining.
        """
        if not full_connect:
            yield from self.configiter()
        else:
            yield from self.configiter_pretrain()

    def configiter(self):
        """Sample a configuration from this world."""
        while True:
            with torch.no_grad():
                gammaexp = self.gamma.sigmoid()
                gammaexp = torch.empty_like(gammaexp).uniform_().lt_(gammaexp)
                gammaexp.diagonal().zero_()
            yield gammaexp

    def configiter_pretrain(self):
        """Sample a configuration from this world."""
        while True:
            with torch.no_grad():
                gammaexp = torch.ones_like(self.gamma)
                gammaexp.diagonal().zero_()
            yield gammaexp

    def sampleiter_pretrain(self, bs=1):
        return torch.ones([bs, self.M, self.N])

    def sampleiter(self, bs=1):
        """
        Ancestral sampling with MLP.
        
        1 sample is a tensor (1, M, N).
        A minibatch of samples is a tensor (bs, M, N).
        1 variable is a tensor (bs, 1, N)
        """
        while True:
            with torch.no_grad():
                h = []   # Hard (onehot) samples  (bs,1,N)
                for i in range(self.M):
                    O = torch.zeros(bs, self.M-i, self.N)   # (bs,M-i,N)
                    v = torch.cat(h+[O], dim=1)             # (bs,M-i,N) + (bs,1,N)*i
                    v = torch.einsum("hik,i,bik->bh", self.W0gt[i], self.gammagt[i], v)
                    v = v + self.B0gt[i].unsqueeze(0)
                    v = self.leaky_relu(v)
                    v = torch.einsum("oh,bh->bo",     self.W1gt[i], v)
                    v = v + self.B1gt[i].unsqueeze(0)
                    v = v.softmax(dim=1).unsqueeze(1)
                    h.append(OneHotCategorical(v).sample())
                s = torch.cat(h, dim=1)
            yield s
    
    def logits(self, sample, config, traingt=False):
        """
        Logits of sample variables given sampled configuration.
        input  sample = (bs, M, N)  # Actual value of the sample
        input  config = (M, M)      # Configuration
        return logits = (bs, M, N)
        """
        W0 = self.W0gt if traingt else self.W0fast+self.W0slow
        W1 = self.W1gt if traingt else self.W1fast+self.W1slow
        B0 = self.B0gt if traingt else self.B0fast+self.B0slow
        B1 = self.B1gt if traingt else self.B1fast+self.B1slow
        
        v = torch.einsum("ihjk,ij,bjk->bih", W0, config, sample)
        v = v + B0.unsqueeze(0)
        v = self.leaky_relu(v)
        v = torch.einsum("ioh,bih->bio",     W1, v)
        v = v + B1.unsqueeze(0)
        return v
    
    def logprob(self, sample, config, block=(), traingt=False):
        """
        Log-probability of sample variables given sampled configuration.
        input  sample = (bs, M, N)  # Actual value of the sample
        input  config = (M, M)      # Configuration
        return logprob = (bs, M)
        """
        
        block = [block] if isinstance(block, int) else list(set(iter(block)))
        block = torch.as_tensor(block, dtype=torch.long, device=sample.device)
        block = torch.ones(self.M, device=sample.device).index_fill_(0, block, 0)
        v = self.logits(sample, config, traingt=traingt) / self.a.temperature
        v = v.log_softmax(dim=2)
        v = torch.einsum("bio,bio->bi", v, sample)
        vn = torch.einsum("bi,i->bi", v, 0+block)
        vi = torch.einsum("bi,i->bi", v, 1-block)
        return vn, vi
    
    def forward(self, sample, config, block=()):
        """Returns the NLL of the samples under the given configuration"""
        return self.logprob(sample, config, block)
    
    def reconstrain_gamma(self):
        with torch.no_grad():
            self.gamma.clamp_(-5,+5)
            self.gamma.diagonal().fill_(float("-inf"))
    
    def reconstrain_theta(self, force=False):
        if force or self.a.structural_only:
            if self.Hlr == self.Hgt:
                # --structural-* flags are only meaningful if the ground-truth
                # and learner are of the same architecture.
                with torch.no_grad():
                    self.W0slow.copy_(self.W0gt)
                    self.W1slow.copy_(self.W1gt)
                    self.B0slow.copy_(self.B0gt)
                    self.B1slow.copy_(self.B1gt)
    
    def parameters_gt(self):
        return iter([self.W0gt, self.B0gt, self.W1gt, self.B1gt])
    def parameters_fastslow(self):
        return zip(iter([self.W0fast, self.B0fast, self.W1fast, self.B1fast]),
                   iter([self.W0slow, self.B0slow, self.W1slow, self.B1slow]))
    def parameters_fast(self):
        for f,s in self.parameters_fastslow(): yield f
    def parameters_slow(self):
        for f,s in self.parameters_fastslow(): yield s
    def parameters(self):
        for f,s in self.parameters_fastslow(): yield f+s
    def zero_fastparams(self):
        with torch.no_grad():
            for f in self.parameters_fast(): f.zero_()



class AsiaWorld(CategoricalWorld):
    def init(self, *args, **kwargs):
        self.initgraph()
        
        self.register_buffer   ("table_asia_gt",   torch.zeros(2))
        self.register_buffer   ("table_tub_gt",    torch.zeros(2,2))
        self.register_buffer   ("table_smoke_gt",  torch.zeros(2))
        self.register_buffer   ("table_lung_gt",   torch.zeros(2,2))
        self.register_buffer   ("table_bronc_gt",  torch.zeros(2,2))
        self.register_buffer   ("table_either_gt", torch.zeros(2,2,2))
        self.register_buffer   ("table_xray_gt",   torch.zeros(2,2))
        self.register_buffer   ("table_dysp_gt",   torch.zeros(2,2,2))
        
        self.register_parameter("gamma",   Parameter(torch.zeros_like(self.gammagt)))
        
        self.register_parameter("W0slow",  Parameter(torch.zeros(self.M, self.Hlr, self.M, self.N)))
        self.register_parameter("B0slow",  Parameter(torch.zeros(self.M, self.Hlr                )))
        self.register_parameter("W1slow",  Parameter(torch.zeros(self.M, self.N, self.Hlr        )))
        self.register_parameter("B1slow",  Parameter(torch.zeros(self.M, self.N                  )))
        
        self.register_parameter("W0fast",  Parameter(torch.zeros_like(self.W0slow)))
        self.register_parameter("B0fast",  Parameter(torch.zeros_like(self.B0slow)))
        self.register_parameter("W1fast",  Parameter(torch.zeros_like(self.W1slow)))
        self.register_parameter("B1fast",  Parameter(torch.zeros_like(self.B1slow)))
        
        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=0.1)
        
        for i in range(self.M):
            torch.nn.init.orthogonal_(self.W0slow[i])
            torch.nn.init.orthogonal_(self.W1slow[i])
        torch.nn.init.uniform_(self.B0slow,    -.1, +.1)
        torch.nn.init.uniform_(self.B1slow,    -.1, +.1)
        torch.nn.init.uniform_(self.gamma, -.1, +.1)
        with torch.no_grad(): self.gamma.diagonal().fill_(float("-inf"))
        
        self.alterdists()
        self.reconstrain_theta(force=self.a.structural_init)
    
    def initgraph(self):
        self.M = self.a.num_vars = 8
        self.N = self.a.num_cats = 2
        self.initgammagt()
        self.gammagt.zero_()
        # 0->1->5->6,2->3->5->7,2->4->7
        self.gammagt[1,0] = 1
        self.gammagt[5,1] = 1
        self.gammagt[6,5] = 1
        self.gammagt[3,2] = 1
        self.gammagt[5,3] = 1
        self.gammagt[7,5] = 1
        self.gammagt[4,2] = 1
        self.gammagt[7,4] = 1
    
    def alterdists(self):
        self.table_asia_gt  .copy_(torch.as_tensor([  0.01, 0.99  ]))
        self.table_tub_gt   .copy_(torch.as_tensor([ [0.05, 0.95],
                                                     [0.01, 0.99] ]))
        self.table_smoke_gt .copy_(torch.as_tensor([  0.5,  0.5   ]))
        self.table_lung_gt  .copy_(torch.as_tensor([ [0.1,  0.9 ],
                                                     [0.01, 0.99] ]))
        self.table_bronc_gt .copy_(torch.as_tensor([ [0.6,  0.4 ],
                                                     [0.3,  0.7 ] ]))
        self.table_either_gt.copy_(torch.as_tensor([[[1.0,  0.0 ],
                                                     [1.0,  0.0 ]],
                                                    [[1.0,  0.0 ],
                                                     [0.0,  1.0 ]]]))
        self.table_xray_gt  .copy_(torch.as_tensor([ [0.98, 0.02],
                                                     [0.05, 0.95] ]))
        self.table_dysp_gt  .copy_(torch.as_tensor([[[0.9,  0.1 ],
                                                     [0.7,  0.3 ]],
                                                    [[0.8,  0.2 ],
                                                     [0.1,  0.9 ]]]))
    
    @contextlib.contextmanager
    def intervene(self, *args, **kwargs):
        """Perform an intervention, then undo it."""
        
        class AsiaWorldIntervention:
            def __init__(salf, i=None):
                salf.node = [np.random.randint(0, self.M) if i is None else i]
            
            def __iter__(salf):
                return iter(salf.node)
            
            def do(salf):
                params = list(self.parameters_gt())
                with torch.no_grad():
                    salf.clones = [t.clone() for t in params]
                for i in salf.node:
                    with torch.no_grad():
                        torch.nn.init.uniform_(params[i], -4, +4)
                        params[i].copy_(params[i].softmax(-1))
            
            def undo(salf):
                with torch.no_grad():
                    for tnew, told in zip(self.parameters_gt(), salf.clones):
                        tnew.copy_(told)
        
        intervention = AsiaWorldIntervention(*args, **kwargs)
        try:
            intervention.do()
            yield intervention
        except:  raise
        finally: intervention.undo()
    
    def sampleiter(self, bs=1):
        """Ancestral Sampling from Conditional Probability Tables"""
        while True:
            h = []
            h.append(OneHotCategorical(torch.einsum(        "i->i",  self.table_asia_gt        ))      .sample((bs,)))
            h.append(OneHotCategorical(torch.einsum(    "ai,za->zi", self.table_tub_gt,    h[0]))      .sample())
            h.append(OneHotCategorical(torch.einsum(        "i->i",  self.table_smoke_gt       ))      .sample((bs,)))
            h.append(OneHotCategorical(torch.einsum(    "ai,za->zi", self.table_lung_gt,   h[2]))      .sample())
            h.append(OneHotCategorical(torch.einsum(    "ai,za->zi", self.table_bronc_gt,  h[2]))      .sample())
            h.append(OneHotCategorical(torch.einsum("bai,za,zb->zi", self.table_either_gt, h[1], h[3])).sample())
            h.append(OneHotCategorical(torch.einsum(    "ai,za->zi", self.table_xray_gt,   h[5]))      .sample())
            h.append(OneHotCategorical(torch.einsum("bai,za,zb->zi", self.table_dysp_gt,   h[4], h[5])).sample())
            yield torch.stack(h, dim=1)
    
    def logits(self, sample, config, traingt=False):
        """
        Logits of sample variables given sampled configuration.
        input  sample = (bs, M, N)  # Actual value of the sample
        input  config = (M, M)      # Configuration
        return logits = (bs, M, N)
        """
        assert(not traingt)
        return super().logits(sample, config, traingt)
    
    def parameters_gt(self):
        return iter([self.table_asia_gt,
                     self.table_tub_gt,
                     self.table_smoke_gt,
                     self.table_lung_gt,
                     self.table_bronc_gt,
                     self.table_either_gt,
                     self.table_xray_gt,
                     self.table_dysp_gt,
        ])

