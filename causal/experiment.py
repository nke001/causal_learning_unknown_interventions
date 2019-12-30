# -*- coding: utf-8 -*-
import nauka
import io, logging, os, sys, time, pdb
import torch
import uuid

from   PIL                              import Image
from   zipfile                          import ZipFile

from   .models                          import *


class ExperimentBase(nauka.exp.Experiment):
    """
    The base class for all experiments.
    """
    def __init__(self, a):
        self.a = type(a)(**a.__dict__)
        self.a.__dict__.pop("__argp__", None)
        self.a.__dict__.pop("__argv__", None)
        self.a.__dict__.pop("__cls__",  None)
        if self.a.workDir:
            super().__init__(self.a.workDir)
        else:
            projName = "CausalOptimization-40037046-a359-470b-b327-af9bbef3e532"
            expNames = [] if self.a.name is None else self.a.name
            workDir  = nauka.fhs.createWorkDir(self.a.baseDir, projName, self.uuid, expNames)
            super().__init__(workDir)
        self.mkdirp(self.logDir)
    
    def reseed(self, password=None):
        """
        Reseed PRNGs for reproducibility at beginning of interval.
        """
        password = password or "Seed: {} Interval: {:d}".format(self.a.seed,
                                                                self.S.intervalNum,)
        nauka.utils.random.setstate           (password)
        nauka.utils.numpy.random.set_state    (password)
        nauka.utils.torch.random.manual_seed  (password)
        nauka.utils.torch.cuda.manual_seed_all(password)
        return self
    
    def brk(self, it, max=None):
        for i, x in enumerate(it):
            if self.a.fastdebug and i>=self.a.fastdebug: break
            if max is not None  and i>=max:              break
            yield x
    
    @property
    def uuid(self):
        u = nauka.utils.pbkdf2int(128, self.name)
        u = uuid.UUID(int=u)
        return str(u)
    @property
    def dataDir(self):
        return self.a.dataDir
    @property
    def logDir(self):
        return os.path.join(self.workDir, "logs")
    @property
    def isDone(self):
        return (self.S.epochNum >= self.a.num_epochs or
               (self.a.fastdebug and self.S.epochNum >= self.a.fastdebug))
    @property
    def exitcode(self):
        return 0 if self.isDone else 1



class Experiment(ExperimentBase):
    """
    Causal experiment.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        class MsgFormatter(logging.Formatter):
            def formatTime(self, record, datefmt):
                t           = record.created
                timeFrac    = abs(t-int(t))
                timeStruct  = time.localtime(record.created)
                timeString  = ""
                timeString += time.strftime("%F %T", timeStruct)
                timeString += "{:.3f} ".format(timeFrac)[1:]
                timeString += time.strftime("%Z",    timeStruct)
                return timeString
        formatter = MsgFormatter("[%(asctime)s ~~ %(levelname)-8s] %(message)s")
        handlers = [
            logging.FileHandler  (os.path.join(self.logDir, "log.txt")),
            logging.StreamHandler(sys.stdout),
        ]
        for h in handlers:
            h.setFormatter(formatter)
        logging.basicConfig(
            level    = logging.INFO,
            handlers = handlers,
        )
        logging.info("*****************************************")
        logging.info("Command:             "+" ".join(sys.argv))
        logging.info("CWD:                 "+os.getcwd())
        logging.info("Experiment Name:     "+self.name)
        logging.info("Experiment UUID:     "+self.uuid)
        logging.info("Experiment Work Dir: "+self.workDir)
        logging.info("")
        logging.info("")
    
    @property
    def name(self):
        """A unique name containing every attribute that distinguishes this
        experiment from another and no attribute that does not."""
        attrs = [
            self.a.seed,
            self.a.model,
            self.a.num_epochs,
            self.a.batch_size,
            self.a.dpe,
            self.a.train_functional,
            self.a.ipd,
            self.a.hidden_truth,
            self.a.hidden_learn,
            self.a.num_vars,
            self.a.num_cats,
            self.a.num_parents,
            self.a.cpi,
            self.a.xfer_epi_size,
            self.a.predict,
            self.a.predict_cpb,
            self.a.temperature,
            self.a.structural_only,
            self.a.structural_init,
            self.a.graph,
            self.a.cuda,
            self.a.model_optimizer,
            self.a.gamma_optimizer,
            self.a.lmaxent,
            self.a.lsparse,
            self.a.ldag,
            self.a.fastdebug,
        ]
        return "-".join([str(s) for s in attrs]).replace("/", "_")
    
    def load(self, path):
        self.S = torch.load(os.path.join(path, "snapshot.pkl"))
        return self
    
    def dump(self, path):
        torch.save(self.S,  os.path.join(path, "snapshot.pkl"))
        return self
    
    def fromScratch(self):
        pass
        """Reseed PRNGs for initialization step"""
        self.reseed(password="Seed: {} Init".format(self.a.seed))
        
        """Create snapshottable-state object"""
        self.S = nauka.utils.PlainObject()
        
        """Model Instantiation"""
        self.S.model = None
        if   self.a.model == "cat":
            self.S.model  = CategoricalWorld(self.a)
        elif self.a.model == "asia":
            self.S.model  = AsiaWorld(self.a)
        if   self.S.model is None:
            raise ValueError("Unsupported model \""+self.a.model+"\"!")
        
        if self.a.cuda:
            self.S.model  = self.S.model.cuda(self.a.cuda[0])
        else:
            self.S.model  = self.S.model.cpu()
        
        """Optimizer Selection"""
        self.S.msoptimizer = nauka.utils.torch.optim.fromSpec(self.S.model.parameters_slow(),       self.a.model_optimizer)
        self.S.goptimizer  = nauka.utils.torch.optim.fromSpec(self.S.model.structural_parameters(), self.a.gamma_optimizer)
        
        """Counters"""
        self.S.epochNum    = 0
        self.S.intervalNum = 0
        self.S.stepNum     = 0
        
        return self
    
    def run(self):
        """Run by intervals until experiment completion."""
        while not self.isDone:
            self.interval().snapshot().purge()
        return self
    
    def interval(self):
        """
        An interval is defined as the computation- and time-span between two
        snapshots.
        
        Hard requirements:
        - By definition, one may not invoke snapshot() within an interval.
        - Corollary: The work done by an interval is either fully recorded or
          not recorded at all.
        - There must be a step of the event logger between any TensorBoard
          summary log and the end of the interval.
        
        For reproducibility purposes, all PRNGs are reseeded at the beginning
        of every interval.
        """
        
        self.reseed()
        
        
        """Training Loop"""
        self.S.model.train()
        for q in self.brk(range(self.a.dpe)):
            if q>0: self.S.stepNum += 1
            
            """Initialize a new distribution"""
            self.S.model.alterdists()
            self.S.model.zero_fastparams()
            
            
            """Train functional parameters only Loop"""
            if self.a.train_functional:
                smpiter = self.S.model.sampleiter(self.a.batch_size)
                cfgiter = self.S.model.configpretrainiter()
                for b, (batch, config) in self.brk(enumerate(zip(smpiter, cfgiter)), max=self.a.train_functional):
                    self.S.msoptimizer.zero_grad()
                    nll = -self.S.model.logprob(batch, config)[0].mean()
                    nll.backward()
                    self.S.msoptimizer.step()
                    if self.a.verbose and b % self.a.verbose == 0:
                        logging.info("Train functional param only NLL: "+str(nll.item()))
            
            
            """Interventions Loop"""
            for j in self.brk(range(self.a.ipd)):
                if j>0: self.S.stepNum += 1
                intervention_tstart = time.time()
                
                """Perform intervention under guard."""
                with self.S.model.intervene() as intervention:
                    """Possibly attempt to predict the intervention node,
                       instead of relying on knowledge of it."""
                    if self.a.predict:
                        with torch.no_grad():
                            accnll  = 0
                            smpiter = self.S.model.sampleiter(self.a.batch_size)
                            cfgiter = self.S.model.configpretrainiter()
                            for batch in self.brk(smpiter, max=self.a.predict):
                                for config in self.brk(cfgiter, max=self.a.predict_cpb):
                                    accnll += -self.S.model.logprob(batch, config)[0].mean(0)
                            selnode = torch.argmax(accnll).item()
                            logging.info("Predicted Intervention Node: {}  Actual Intervention Node: {}".format([selnode], list(iter(intervention))))
                            intervention = selnode
                    
                    self.S.goptimizer.zero_grad()
                    self.S.model.gamma.grad = torch.zeros_like(self.S.model.gamma)
                    
                    gammagrads = [] # List of T tensors of shape (M,M,) indexed by (i,j)
                    logregrets = [] # List of T tensors of shape (M,)   indexed by (i,)
                    
                    """Transfer Episode Adaptation Loop"""
                    smpiter = self.S.model.sampleiter(self.a.batch_size)
                    for batch in self.brk(smpiter, max=self.a.xfer_epi_size):
                        gammagrad = 0
                        logregret = 0
                        
                        """Configurations Loop"""
                        cfgiter = self.S.model.configiter()
                        for config in self.brk(cfgiter, max=self.a.cpi):
                            """Accumulate Gamma Gradient"""
                            if self.a.predict:
                                logpn, logpi = self.S.model.logprob(batch, config, block=intervention)
                            else:
                                logpn, logpi = self.S.model.logprob(batch, config)
                            with torch.no_grad():
                                gammagrad += self.S.model.gamma.sigmoid() - config
                                logregret += logpn.mean(0)
                            logpi.sum(1).mean(0).backward()
                        
                        gammagrads.append(gammagrad)
                        logregrets.append(logregret)
                    
                    """Update Fast Optimizer"""
                    for batch in self.brk(smpiter, max=self.a.xfer_epi_size):
                        self.S.model.zero_fastparams()
                        self.S.mfoptimizer = nauka.utils.torch.optim.fromSpec(self.S.model.parameters_fast(), self.a.model_optimizer)
                        self.S.mfoptimizer.zero_grad()
                        cfgiter = self.S.model.configiter()
                        for config in self.brk(cfgiter, max=self.a.cpi):
                            logprob = self.S.model.logprob(batch, config)[0].sum(1).mean()
                            logprob.backward()
                        self.S.mfoptimizer.step()
                    all_logprobs = []
                    for batch in self.brk(smpiter, max=self.a.xfer_epi_size):
                        cfgiter = self.S.model.configiter()
                        for config in self.brk(cfgiter, max=self.a.cpi):
                            all_logprobs.append(self.S.model.logprob(batch, config)[0].mean())
                    
                    
                    """Gamma Gradient Estimator"""
                    with torch.no_grad():
                        gammagrads = torch.stack(gammagrads)
                        logregrets = torch.stack(logregrets)
                        normregret = logregrets.softmax(0)
                        dRdgamma   = torch.einsum("kij,ki->ij", gammagrads, normregret)
                        self.S.model.gamma.grad.copy_(dRdgamma)
                        all_logprobs = torch.stack(all_logprobs).mean()
                    
                    """Gamma Regularizers"""
                    siggamma = self.S.model.gamma.sigmoid()
                    Lmaxent  = ((siggamma)*(1-siggamma)).sum().mul(-self.a.lmaxent)
                    Lsparse  = siggamma.sum().mul(self.a.lsparse)
                    Ldag     = siggamma.mul(siggamma.t()).cosh().tril(-1).sum() \
                                       .sub(self.S.model.M**2 - self.S.model.M) \
                                       .mul(self.a.ldag)
                    (Lmaxent + Lsparse + Ldag).backward()
                    
                    """Perform Gamma Update with constraints"""
                    self.S.goptimizer.step()
                    self.S.model.reconstrain_gamma()
                    
                    """Stop timer"""
                    intervention_tend = time.time()
                    
                    """Print the state of training occasionally"""
                    if self.a.verbose:
                        with torch.no_grad():
                            # Compute Binary Cross-Entropy over gammas, ignoring diagonal
                            siggamma  = self.S.model.gamma.sigmoid()
                            pospred   = siggamma.clone()
                            negpred   = 1-siggamma.clone()
                            posgt     = self.S.model.gammagt
                            neggt     = 1-self.S.model.gammagt
                            pospred.diagonal().fill_(1)
                            negpred.diagonal().fill_(1)
                            bce       = -pospred.log2_().mul_(posgt) -negpred.log2_().mul_(neggt)
                            bce       = bce.sum()
                            bce.div_(siggamma.numel() - siggamma.diagonal().numel())
                            
                            logging.info("")
                            logging.info("**************************")
                            logging.info("Gamma GT:   "+os.linesep+str(self.S.model.gammagt.detach()))
                            logging.info("Gamma:      "+os.linesep+str(siggamma))
                            logging.info("dRdGamma:   "+os.linesep+str(dRdgamma))
                            logging.info("Gamma Grad: "+os.linesep+str(self.S.model.gamma.grad.detach()))
                            logging.info("Gamma CE:   "+str(bce.item()))
                            logging.info("Intervention Time (s):       "+str(intervention_tend-intervention_tstart))
                            logging.info("Exp. temp. Transfer logprob: "+str(all_logprobs.item()))
                            logging.info("")
                            
                            if self.S.stepNum % self.a.verbose == 0:
                                # Append a PNG to a Zip file to avoid too many files
                                # on the filesystem
                                GAMMABIO = io.BytesIO()
                                GAMMAVIZ = self.S.model.vizualize_gamma().numpy()
                                GAMMAIMG = Image.fromarray(GAMMAVIZ, "RGB")
                                GAMMAIMG.save(GAMMABIO, "png")
                                GAMMAPNG = "gamma-{:07d}.png".format(self.S.stepNum)
                                GAMMAZIP = os.path.join(self.logDir, "gamma.zip")
                                with ZipFile(GAMMAZIP, 'a') as GAMMAZIP:
                                    GAMMAZIP.writestr(GAMMAPNG, GAMMABIO.getvalue())
        
        
        """Exit"""
        logging.info("Epoch {:d} done.\n".format(self.S.epochNum))
        self.S.epochNum    += 1
        self.S.intervalNum += 1
        self.S.stepNum     += 1
        return self
