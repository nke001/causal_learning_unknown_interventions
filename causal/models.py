import contextlib
import itertools
import os, sys, time, pdb
import numpy                                as np
import torch
import torch.nn

import causal

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
        
        # Custom-allocated PyTorch tensors
        W0,B0,W1,B1 = self._allocate_torch_weightset("gt")
        self.register_buffer   ("W0gt",    W0)
        self.register_buffer   ("B0gt",    B0)
        self.register_buffer   ("W1gt",    W1)
        self.register_buffer   ("B1gt",    B1)
        W0,B0,W1,B1 = self._allocate_torch_weightset("slow")
        self.register_parameter("W0slow",  Parameter(W0))
        self.register_parameter("B0slow",  Parameter(B0))
        self.register_parameter("W1slow",  Parameter(W1))
        self.register_parameter("B1slow",  Parameter(B1))
        self.zero_grad()
        
        self.register_parameter("gamma",   Parameter(torch.zeros_like(self.gammagt)))
        
        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=0.1)
        
        for i in range(self.M):
            torch.nn.init.orthogonal_(self.W0slow[i].transpose(0,1))
            torch.nn.init.orthogonal_(self.W1slow[self.Nc[i]:self.Nc[i]+self.N[i]])
        torch.nn.init.uniform_(self.B0slow, -.1, +.1)
        torch.nn.init.uniform_(self.B1slow[self.Nc[i]:self.Nc[i]+self.N[i]], -.1, +.1)
        torch.nn.init.uniform_(self.gamma,  -.1, +.1)
        with torch.no_grad(): self.gamma.diagonal().fill_(float("-inf"))
        
        self.alterdists()
        if self.a.summary:
            print(self.gammagt)
            sys.exit()
    
    @property
    def Nm(self):
        return int(max(self.N))
    
    @property
    def Ns(self):
        return int(sum(self.N))
    
    @property
    def Nc(self):
        return self.N.cumsum().astype(self.N.dtype) - self.N
    
    @property
    def Hgt(self):
        if self.a.hidden_truth is None:
            if self.M > self.Nm: return 4*self.M
            else:                return 4*self.Nm
        else:
            return int(self.a.hidden_truth)
    
    @property
    def Hlr(self):
        if self.a.hidden_learn is None:
            if self.M > self.Nm: return 4*self.M
            else:                return 4*self.Nm
        else:
            return int(self.a.hidden_learn)
    
    @property
    def _np_Hgtu8(self):
        return (self.Hgt+7)&~7
    
    @property
    def _np_Hlru8(self):
        return (self.Hlr+7)&~7
    
    def _allocate_torch_weightset(self, weightset="slow"):
        H   = self.Hlr       if weightset in {"slow"} else self.Hgt
        Hu8 = self._np_Hlru8 if weightset in {"slow"} else self._np_Hgtu8
        M   = self.M
        N   = self.Nm
        Ns  = self.Ns
        W0shape, W0stride = (M,Ns,H), (Ns*Hu8, Hu8, 1)
        B0shape, B0stride = (M,H),    (Hu8,    1)
        W1shape, W1stride = (Ns,H),   (Hu8,    1)
        B1shape, B1stride = (Ns,),    (1,)
        W0  = torch.empty_strided(W0shape, W0stride, dtype=torch.float32, device="cpu")
        B0  = torch.empty_strided(B0shape, B0stride, dtype=torch.float32, device="cpu")
        W1  = torch.empty_strided(W1shape, W1stride, dtype=torch.float32, device="cpu")
        B1  = torch.empty_strided(B1shape, B1stride, dtype=torch.float32, device="cpu")
        W0s = W0.storage()
        B0s = B0.storage()
        W1s = W1.storage()
        B1s = B1.storage()
        W0s.resize_((W0s.size()+7)&~7).fill_(0)
        B0s.resize_((B0s.size()+7)&~7).fill_(0)
        W1s.resize_((W1s.size()+7)&~7).fill_(0)
        B1s.resize_((B1s.size()+7)&~7).fill_(0)
        return W0,B0,W1,B1
    
    def _convert_to_np_weightset(self, *weightset):
        """
        Convert PyTorch weightsets to Numpy weightsets
        
        We are obliged to do this because of the particularities of the C code,
        and because PyTorch Tensor fails to implement the buffer protocol.
        """
        W0,B0,W1,B1 = weightset
        W0 = W0.detach().numpy()
        B0 = B0.detach().numpy()
        W1 = W1.detach().numpy()
        B1 = B1.detach().numpy()
        return W0,B0,W1,B1
    
    def zero_grad(self):
        # Custom-allocated zero-gradient tensors.
        self.W0gt.grad,   self.B0gt.grad,   self.W1gt.grad,   self.B1gt.grad   = self._allocate_torch_weightset("gt")
        self.W0slow.grad, self.B0slow.grad, self.W1slow.grad, self.B1slow.grad = self._allocate_torch_weightset("slow")
        
        # Numpy views onto PyTorch tensors.
        self._np_W0gt,    self._np_B0gt,    self._np_W1gt,    self._np_B1gt    = self._convert_to_np_weightset(self.W0gt,        self.B0gt,        self.W1gt,        self.B1gt)
        self._np_W0slow,  self._np_B0slow,  self._np_W1slow,  self._np_B1slow  = self._convert_to_np_weightset(self.W0slow,      self.B0slow,      self.W1slow,      self.B1slow)
        self._np_dW0gt,   self._np_dB0gt,   self._np_dW1gt,   self._np_dB1gt   = self._convert_to_np_weightset(self.W0gt.grad,   self.B0gt.grad,   self.W1gt.grad,   self.B1gt.grad)
        self._np_dW0slow, self._np_dB0slow, self._np_dW1slow, self._np_dB1slow = self._convert_to_np_weightset(self.W0slow.grad, self.B0slow.grad, self.W1slow.grad, self.B1slow.grad)
    
    def initgraph(self):
        if self.a.graph is None:
            self.M = self.a.num_vars
            self.N = np.asarray([self.a.num_cats]*self.M, dtype=np.int32)
            self.initgammagt()
            
            expParents = self.a.num_parents
            idx        = np.arange(self.M).astype(np.float32)[:,np.newaxis]
            idx_maxed  = np.minimum(idx*0.5, expParents)
            p          = np.broadcast_to(idx_maxed/(idx+1), (self.M, self.M))
            B          = np.random.binomial(1, p)
            B          = np.tril(B, -1)
            self.gammagt.copy_(torch.as_tensor(B))
        elif len(self.a.graph) == 1 and os.path.isfile(self.a.graph[0]):
            self.bif = causal.bif.load(self.a.graph[0])
            self.bif.thermalize(self.a.temperature_gt,
                                self.a.temperature_pthresh)
            N = [v.num_choices for v in self.bif.var_list]
            self.M = self.bif.num_variables
            self.N = np.asarray(N, dtype=np.int32)
            self.initgammagt()
            self.gammagt.copy_(torch.from_numpy(self.bif.adjacency_matrix()))
        else:
            gammagt = causal.utils.parse_skeleton(self.a.graph, self.a.num_vars)
            N = [self.a.num_cats]*self.a.num_vars
            self.M = self.a.num_vars
            self.N = np.asarray(N, dtype=np.int32)
            self.initgammagt()
            self.gammagt.copy_(torch.from_numpy(gammagt))
        
        return self
    
    def initgammagt(self):
        if not hasattr(self, "gammagt"):
            self.register_buffer("gammagt", torch.empty((self.M, self.M),
                                                        dtype=torch.float32))
        self.gammagt.zero_()
    
    def initweightset(self, i, *weightset, gain=1, a=0, b=1):
        W0,B0,W1,B1 = self.sliceweightset(i, *weightset)
        torch.nn.init.orthogonal_(W0.transpose(0,1), gain=gain)
        torch.nn.init.orthogonal_(W1,                gain=gain)
        torch.nn.init.uniform_(B0, a=a, b=b)
        torch.nn.init.uniform_(B1, a=a, b=b)
        return W0,B0,W1,B1
    
    def sliceweightset(self, i, *weightset):
        W0,B0,W1,B1 = weightset
        Nc = self.Nc
        N  = self.N
        W0 = W0[i]
        B0 = B0[i]
        W1 = W1[Nc[i]:Nc[i]+N[i]]
        B1 = B1[Nc[i]:Nc[i]+N[i]]
        return W0,B0,W1,B1
    
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
            torch.nn.init.orthogonal_(self.W0gt[i].transpose(0,1), 2.5)
            torch.nn.init.orthogonal_(self.W1gt[self.Nc[i]:self.Nc[i]+self.N[i]], 2.5)
            torch.nn.init.uniform_(self.B0gt[i], -1.1, +1.1)
            torch.nn.init.uniform_(self.B1gt[self.Nc[i]:self.Nc[i]+self.N[i]], -1.1, +1.1)
        return self
    
    def intervention(self, *args, **kwargs):
        """Create an intervention."""
        
        class CategoricalWorldIntervention:
            def __init__(salf, i=None, dataset=None):
                salf.node    = [np.random.randint(0, self.M) if i is None else i]
                salf.dataset = dataset
            def __iter__(salf):
                return iter(salf.node)
            def __enter__(salf):
                raise NotImplementedError
            def __exit__(salf, *args, **kwargs):
                raise NotImplementedError
        
        class CategoricalWorldInterventionMLP(CategoricalWorldIntervention):
            def __init__(salf, *args, **kwargs):
                """Construct an intervention over MLPs"""
                super().__init__(*args, **kwargs)
                salf.mods = []
                with torch.no_grad():
                    for i in salf.node:
                        W0 = self.W0gt[i].clone()
                        W1 = self.W1gt[self.Nc[i]:self.Nc[i]+self.N[i]].clone()
                        B0 = self.B0gt[i].clone()
                        B1 = self.B1gt[self.Nc[i]:self.Nc[i]+self.N[i]].clone()
                        torch.nn.init.orthogonal_(W0.transpose(0,1), 2.5)
                        torch.nn.init.orthogonal_(W1, 2.5)
                        torch.nn.init.uniform_(B0, -1.1, +1.1)
                        torch.nn.init.uniform_(B1, -1.1, +1.1)
                        salf.mods.append((i,W0,W1,B0,B1))
                    
                    if self.a.limit_samples > 0 and not salf.dataset:
                        with salf: # Apply ourselves!
                            salf.dataset = next(self.sampleiter(self.a.limit_samples))
            
            def __enter__(salf):
                """Apply this intervention"""
                with torch.no_grad():
                    self.dataset = salf.dataset
                    salf.clones = [t.clone() for t in self.parameters_gt()]
                    for i,W0,W1,B0,B1 in salf.mods:
                        self.W0gt[i].copy_(W0)
                        self.W1gt[self.Nc[i]:self.Nc[i]+self.N[i]].copy_(W1)
                        self.B0gt[i].copy_(B0)
                        self.B1gt[self.Nc[i]:self.Nc[i]+self.N[i]].copy_(B1)
                    return salf
            def __exit__(salf, *args, **kwargs):
                """Unapply this intervention"""
                with torch.no_grad():
                    self.dataset = None
                    for tnew, told in zip(self.parameters_gt(), salf.clones):
                        tnew.copy_(told)
        
        class CategoricalWorldInterventionCPT(CategoricalWorldIntervention):
            def __init__(salf, *args, **kwargs):
                super().__init__(*args, **kwargs)
                salf.mods = []
                with torch.no_grad():
                    for i in salf.node:
                        i = int(i)
                        v = self.bif[i]
                        t = torch.from_numpy(v.cpt).clone()
                        torch.nn.init.uniform_(t, -2.5, +2.5)
                        t.copy_(t.softmax(dim=-1))
                        salf.mods.append((i,t))
                    if self.a.limit_samples > 0 and not salf.dataset:
                        with salf: # Apply ourselves!
                            salf.dataset = next(self.sampleiter(self.a.limit_samples))
            
            def __enter__(salf):
                with torch.no_grad():
                    self.dataset = salf.dataset
                    salf.buffer_clone = self.bif.buffer.copy()
                    for i,t in salf.mods:
                        self.bif[i].cpt[:] = t.numpy()
                    return salf
                
            def __exit__(salf, *args, **kwargs):
                with torch.no_grad():
                    self.dataset       = None
                    self.bif.buffer[:] = salf.buffer_clone
        
        if hasattr(self, "bif"):
            return CategoricalWorldInterventionCPT(*args, **kwargs)
        else:
            return CategoricalWorldInterventionMLP(*args, **kwargs)
    
    def configiter(self):
        """Sample a configuration from this world."""
        while True:
            with torch.no_grad():
                gammaexp = self.gamma.sigmoid()
                gammaexp = torch.empty_like(gammaexp).uniform_().lt_(gammaexp)
                gammaexp.diagonal().zero_()
            yield gammaexp

    def sampleiter(self, bs=None):
        """
        Ancestral sampling with MLP.
        
        A minibatch of samples is a tensor (M, bs).
        """
        
        if bs is None:
            bs = self.a.batch_size
        
        if hasattr(self, "dataset") and self.dataset is not None:
            while True:
                i = torch.randint(int(self.dataset.shape[1]), (int(bs),))
                yield self.dataset.index_select(1, i).contiguous()
        
        if hasattr(self, "bif"):
            while True:
                s = np.empty((self.M, bs), dtype=np.int32)
                causal._causal.sample_cpt(self.bif, out=s)
                yield torch.from_numpy(s)
        else:
            while True:
                s = np.empty((self.M, bs), dtype=np.int32)
                causal._causal.sample_mlp(self._np_W0gt, self._np_B0gt,
                                          self._np_W1gt, self._np_B1gt,
                                          self.N,        self.gammagt.numpy(),
                                          out=s,         alpha=0.1)
                yield torch.from_numpy(s)
    
    def logprob_nograd(self, sample, config, block=(), weightset="slow"):
        block  = [block] if isinstance(block, int) else list(set(iter(block)))
        mask   = np.zeros((self.M,), dtype=np.float32)
        mask[block] = 1
        block  = mask
        
        sample = sample.numpy()
        config = config.numpy()
        logp   = np.empty((self.M, sample.shape[1]), dtype=np.float32)
        
        if   weightset=="slow":
            causal._causal.logprob_mlp(self._np_W0slow,
                                       self._np_B0slow,
                                       self._np_W1slow,
                                       self._np_B1slow,
                                       self.N,
                                       block,sample,config,logp,
                                       alpha=0.1,temp=self.a.temperature)
        else:
            causal._causal.logprob_mlp(self._np_W0gt,
                                       self._np_B0gt,
                                       self._np_W1gt,
                                       self._np_B1gt,
                                       self.N,
                                       block,sample,config,logp,
                                       alpha=0.1,temp=self.a.temperature)
        
        return torch.from_numpy(logp*(1-block[:,np.newaxis]))
    
    def logprob_simplegrad(self, sample, config, block=(), weightset="slow"):
        block  = [block] if isinstance(block, int) else list(set(iter(block)))
        mask   = np.zeros((self.M,), dtype=np.float32)
        mask[block] = 1
        block  = mask
        
        sample = sample.numpy()
        config = config.numpy()
        logp   = np.empty((self.M, sample.shape[1]), dtype=np.float32)
        
        if   weightset=="slow":
           causal._causal.logprob_mlp(self._np_W0slow,
                                      self._np_B0slow,
                                      self._np_W1slow,
                                      self._np_B1slow,
                                      self.N,
                                      block,sample,config,logp,
                                      self._np_dW0slow,
                                      self._np_dB0slow,
                                      self._np_dW1slow,
                                      self._np_dB1slow,
                                      alpha=0.1,temp=self.a.temperature)
        else:
           causal._causal.logprob_mlp(self._np_W0gt,
                                      self._np_B0gt,
                                      self._np_W1gt,
                                      self._np_B1gt,
                                      self.N,
                                      block,sample,config,logp,
                                      self._np_dW0gt,
                                      self._np_dB0gt,
                                      self._np_dW1gt,
                                      self._np_dB1gt,
                                      alpha=0.1,temp=self.a.temperature)
        
        return torch.from_numpy(logp*(1-block[:,np.newaxis]))
    
    def forward(self, sample, config, block=()):
        """Returns the NLL of the samples under the given configuration"""
        return self.logprob_nograd(sample, config, block)
    
    def reconstrain_gamma(self):
        with torch.no_grad():
            self.gamma.clamp_(-5,+5)
            self.gamma.diagonal().fill_(float("-inf"))
    
    def parameters_gt(self):
        return iter([self.W0gt, self.B0gt, self.W1gt, self.B1gt])
    def parameters(self):
        return iter([self.W0slow, self.B0slow, self.W1slow, self.B1slow])

