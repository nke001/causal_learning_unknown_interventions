import nauka
import io, logging, os, sys, time, pdb
import torch
import uuid
import causal

from   PIL                              import Image
from   zipfile                          import ZipFile

from   .models                          import *


class ExperimentBase(nauka.exp.Experiment):
    """
    The base class for all experiments.
    """
    def __init__(self, a):
        #
        # Clone a into self.a
        #
        self.a = type(a)(**a.__dict__)
        self.a.__dict__.pop("__argp__", None)
        self.a.__dict__.pop("__argv__", None)
        self.a.__dict__.pop("__cls__",  None)
        
        #
        # Certain influential flags completely override others. It would be
        # best not to leave the overriden flags stale, because they propagate
        # into the workdir filename. So we're obliged to make an early read
        # here.
        #
        if isinstance(self.a.graph, list):
            if len(self.a.graph) == 1 and os.path.isfile(self.a.graph[0]):
                G = causal.bif.load(self.a.graph[0])
                self.a.num_vars    = G.num_variables
                self.a.num_cats    = G.max_indegree
                self.a.num_parents = G.max_indegree
            else:
                M = causal.utils.parse_skeleton(self.a.graph)
                self.a.num_vars    = max(self.a.num_vars, len(M))
                self.a.num_parents = int(M.sum(1).max())
        
        #
        # Create the working directory.
        #
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
        causal._causal.seed(nauka.utils.pbkdf2int(64, password))
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
        with open(os.path.join(self.workDir, "name.txt"), "w") as f:
            f.write("Name: ")
            f.write(self.name)
            f.write("\n")
            f.write("Args: ")
            f.write(" ".join(sys.argv))
            f.write("\n")
    
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
            self.a.temperature_gt,
            self.a.temperature_pthresh,
            self.a.graph,
            self.a.cuda,
            self.a.model_optimizer,
            self.a.gamma_optimizer,
            self.a.lmaxent,
            self.a.lsparse,
            self.a.ldag,
            self.a.limit_interventions,
            self.a.limit_samples,
            self.a.fastdebug,
        ]
        return "-".join([str(s) for s in attrs]).replace("/", "_")
    
    @property
    def uuid(self):
        def stringify_opt(opt):
            name = "{}:lr={}".format(opt.name, opt.lr)
            for attr in ["rho", "mom", "beta", "beta1", "beta2", "curvWW", "eps", "nesterov"]:
                if hasattr(opt, attr): name += ",{}={}".format(attr, getattr(opt, attr))
            return name
            
        s = "M{}_N{}_T{}_predict{}_BS{}_seed{}_lsparse{}".format(
            self.a.num_vars,
            self.a.num_cats,
            self.a.temperature,
            self.a.predict,
            self.a.batch_size,
            self.a.seed,
            self.a.lsparse,
        )
        return s+"__UUID-"+super().uuid
    
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
        if   self.S.model is None:
            raise ValueError("Unsupported model \""+self.a.model+"\"!")
        
        """Optimizer Selection"""
        self.S.msoptimizer = nauka.utils.torch.optim.fromSpec(self.S.model.parameters(),            self.a.model_optimizer)
        self.S.goptimizer  = nauka.utils.torch.optim.fromSpec(self.S.model.structural_parameters(), self.a.gamma_optimizer)
        
        
        """Regularizer Annealing"""
        self.S.lsparse     = nauka.utils.lr.fromSpecList(self.a.lsparse)
        
        
        """Counters"""
        self.S.epochNum    = 0
        self.S.intervalNum = 0
        self.S.stepNum     = 0
        
        return self
    
    def readyDataset(self):
        """Prepare data (interventions) reproducibly"""
        self.reseed(password="Seed: {} Data".format(self.a.seed))
        
        if self.a.limit_interventions > 0:
            if not hasattr(self, "interventions_superlist"):
                self.interventions_superlist = [
                    [
                        self.S.model.intervention(i=i)
                        for _ in range(self.a.limit_interventions)
                    ]
                    for i in range(self.S.model.M)
                ]
    
    def run(self):
        """Run by intervals until experiment completion."""
        if self.a.pdb: pdb.set_trace()
        self.readyDataset()
        while not self.isDone:
            self.interval().snapshot().purge()
        return self
    
    def intervention(self):
        """Select intervention, randomly sampled or from a limited set."""
        if self.a.limit_interventions > 0:
            i = int(torch.randint(len(self.interventions_superlist),    (1,)))
            j = int(torch.randint(len(self.interventions_superlist[i]), (1,)))
            return self.interventions_superlist[i][j]
        else:
            return self.S.model.intervention()
    
    def get_status_report(self, siggamma, thresh=0.5):
        gammagt = self.S.model.gammagt
        assert siggamma.size() == gammagt.size()
        
        fp = 0
        fn = 0
        tp = 0
        
        # Print Header
        s  = "".rjust(8) + " "
        for j in range(0, siggamma.shape[0], 4):
            s += str(j).ljust(8)
        s += os.linesep
        
        # Print Body
        for i in range(siggamma.shape[0]):
            s += str(i).rjust(8)
            for j in range(siggamma.shape[1]):
                sig = siggamma[i,j] >= thresh
                gnd = bool(gammagt[i,j])
                tp += 1 if sig and     gnd else 0
                fp += 1 if sig and not gnd else 0
                fn += 1 if not sig and gnd else 0
                if   not sig and not gnd: s += "  "
                elif not sig and     gnd: s += " #"
                elif     sig and not gnd: s += " ."
                else:                     s += " *"
            s += "|"
            if hasattr(self.S.model, "bif"):
                s += "  "+self.S.model.bif[i].name
            s += os.linesep
        s += "True edge: *        False positive: .        False negative: #" + os.linesep
        s += "Hamming Distance @ Threshold "+str(thresh)+":   {:d} (+{:d}-{:d}={:d})".format(fp+fn,fp,fn,tp)
        
        # Return
        return s, fp+fn
    
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
        self.S.model.zero_grad()
        for q in self.brk(range(self.a.dpe)):
            if q>0: self.S.stepNum += 1
            
            """Initialize a new distribution"""
            self.S.model.alterdists()
            
            
            """Train functional parameters only Loop"""
            if self.a.train_functional:
                smpiter = self.S.model.sampleiter()
                cfgiter = self.S.model.configiter()
                for b, (batch, config) in self.brk(enumerate(zip(smpiter, cfgiter)), max=self.a.train_functional):
                    self.S.msoptimizer.zero_grad()
                    nll = -self.S.model.logprob_simplegrad(batch, config).mean()
                    #nll.backward()
                    self.S.model.W0slow.grad.mul_(-1.0/self.S.model.M)
                    self.S.model.B0slow.grad.mul_(-1.0/self.S.model.M)
                    self.S.model.W1slow.grad.mul_(-1.0/self.S.model.M)
                    self.S.model.B1slow.grad.mul_(-1.0/self.S.model.M)
                    self.S.msoptimizer.step()
                    if self.a.verbose and b % self.a.verbose == 0:
                        logging.info("Train functional param only NLL: "+str(nll.item()))
            
            
            """Interventions Loop"""
            for j in self.brk(range(self.a.ipd)):
                if j>0: self.S.stepNum += 1
                intervention_tstart = time.time()
                
                """Perform intervention under guard."""
                with self.intervention() as intervention:
                    """Possibly attempt to predict the intervention node,
                       instead of relying on knowledge of it."""
                    if self.a.predict:
                        with torch.no_grad():
                            accnll  = 0
                            smpiter = self.S.model.sampleiter()
                            cfgiter = self.S.model.configiter()
                            for batch in self.brk(smpiter, max=self.a.predict):
                                for config in self.brk(cfgiter, max=self.a.predict_cpb):
                                    accnll -= self.S.model.logprob_nograd(batch, config).mean(1)
                            selnode = torch.argmax(accnll).item()
                            logging.info("Predicted Intervention Node: {}  Actual Intervention Node: {}".format([selnode], list(iter(intervention))))
                            intervention = selnode
                    
                    self.S.goptimizer.zero_grad()
                    self.S.model.gamma.grad = torch.zeros_like(self.S.model.gamma)
                    
                    gammagrads = [] # List of T tensors of shape (M,M,) indexed by (i,j)
                    logregrets = [] # List of T tensors of shape (M,)   indexed by (i,)
                    
                    """Transfer Episode Adaptation Loop"""
                    with torch.no_grad():
                        gammasigmoid = self.S.model.gamma.sigmoid()
                        smpiter = self.S.model.sampleiter()
                        for batch in self.brk(smpiter, max=self.a.xfer_epi_size):
                            gammagrad = 0
                            logregret = 0
                            
                            """Configurations Loop"""
                            cfgiter = self.S.model.configiter()
                            for config in self.brk(cfgiter, max=self.a.cpi):
                                """Accumulate Gamma Gradient"""
                                logpn = self.S.model.logprob_nograd(batch, config, block=intervention)
                                gammagrad += gammasigmoid - config
                                logregret += logpn.mean(1)
                            
                            gammagrads.append(gammagrad)
                            logregrets.append(logregret)
                    
                    """Gamma Gradient Estimator"""
                    with torch.no_grad():
                        gammagrads = torch.stack(gammagrads)
                        logregrets = torch.stack(logregrets)
                        normregret = logregrets.softmax(0)
                        dRdgamma   = torch.einsum("kij,ki->ij", gammagrads, normregret)
                        self.S.model.gamma.grad.copy_(dRdgamma)
                    
                    """Gamma Regularizers"""
                    siggamma = self.S.model.gamma.sigmoid()
                    Lmaxent  = ((siggamma)*(1-siggamma)).sum().mul(-self.a.lmaxent)
                    Lsparse  = siggamma.sum().mul(float(self.S.lsparse))
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
                            
                            # Compute report
                            report = self.get_status_report(siggamma)[0]
                            
                            logging.info("")
                            logging.info("**************************")
                            logging.info("Gamma:      "+os.linesep+str(siggamma))
                            logging.info("Gamma GT:   "+os.linesep+report)
                            logging.info("Gamma CE:   "+str(bce.item()))
                            logging.info("Intervention Time (s):       "+str(intervention_tend-intervention_tstart))
                            logging.info("Lsparse regularizer:         "+str(float(self.S.lsparse)))
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
                    
                    """Anneal lsparse."""
                    self.S.lsparse.step()
        
        
        """Exit"""
        logging.info("Epoch {:d} done.\n".format(self.S.epochNum))
        self.S.epochNum    += 1
        self.S.intervalNum += 1
        self.S.stepNum     += 1
        return self
