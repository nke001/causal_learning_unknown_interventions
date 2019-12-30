#!/usr/bin/env python
# -*- coding: utf-8 -*-
# PYTHON_ARGCOMPLETE_OK
import pdb, nauka, os, sys


class root(nauka.ap.Subcommand):
    class train(nauka.ap.Subcommand):
        @classmethod
        def addArgs(kls, argp):
            mtxp = argp.add_mutually_exclusive_group()
            mtxp.add_argument("-w", "--workDir",        default=None,         type=str,
                help="Full, precise path to an experiment's working directory.")
            mtxp.add_argument("-b", "--baseDir",        action=nauka.ap.BaseDir)
            argp.add_argument("-d", "--dataDir",        action=nauka.ap.DataDir)
            argp.add_argument("-t", "--tmpDir",         action=nauka.ap.TmpDir)
            argp.add_argument("-n", "--name",           default=[],
                action="append",
                help="Build a name for the experiment.")
            argp.add_argument("-s", "--seed",           default=0,            type=int,
                help="Seed for PRNGs. Default is 0.")
            argp.add_argument("--model",                default="cat",        type=str,
                choices=["cat", "asia"],
                help="Model Selection.")
            argp.add_argument("-e", "--num-epochs",     default=200,          type=int,
                help="Number of epochs")
            argp.add_argument("--batch-size", "--bs",   default=256,          type=int,
                help="Batch Size")
            argp.add_argument("--dpe",                  default=1000,         type=int,
                help="Number of training distributions per epoch")
            argp.add_argument("--train_functional",     default=0,            type=int,
                help="Number of training batches for functional parameters per distribution")
            argp.add_argument("--ipd",                  default=100,          type=int,
                help="Number of interventions per distribution")
            argp.add_argument("--hidden-truth",         default=None,         type=int,
                help="Number of hidden neurons in ground-truth network.")
            argp.add_argument("--hidden-learn",         default=None,         type=int,
                help="Number of hidden neurons in learner network.")
            argp.add_argument("-M", "--num-vars",       default=5,            type=int,
                help="Number of variables in system")
            argp.add_argument("-N", "--num-cats",       default=3,            type=int,
                help="Number of categories per variable, for categorical models")
            argp.add_argument("-P", "--num-parents",    default=5,            type=int,
                help="Number of expected parents. Default is 5.")
            argp.add_argument("-R", "--cpi",            default=20,           type=int,
                help="Configurations per intervention")
            argp.add_argument("-T", "--xfer-epi-size",  default=10,           type=int,
                help="Transfer episode size")
            argp.add_argument("--predict",              default=0,            type=int,
                help="Whether to predict which node was intervened on or not, and "
                     "how many prediction iterations if so.")
            argp.add_argument("--predict-cpb",          default=10,           type=int,
                help="Configurations per batch during intervention prediction.")
            argp.add_argument("--temperature",          default=1.0,          type=float,
                help="Temperature of the MLP. Temperatures > 1 lead to more uniform sampling, temperatures < 1 lead to less uniform sampling.")
            mtxp = argp.add_mutually_exclusive_group()
            mtxp.add_argument("--structural-only",      action="store_true",
                help="Learn structural parameters only, locking the functional parameters to ground truth.")
            mtxp.add_argument("--structural-init",      action="store_true",
                help="Initialize structural parameters to ground truth, but don't lock them.")
            argp.add_argument("-v", "--verbose",        default=0,            type=int,
                nargs="?",   const=10,
                help="Printing interval")
            argp.add_argument("--cuda",                 action=nauka.ap.CudaDevice)
            argp.add_argument("--graph",                action="append",      type=str,
                default=None,
                help="Graph string.")
            argp.add_argument("-p", "--preset",         action=nauka.ap.Preset,
                choices={"blank3":      ["-M", "3", "--graph", ""],
                         "chain3":      ["-M", "3", "--graph", "0->1->2"],
                         "fork3":       ["-M", "3", "--graph", "0->1,0->2"],
                         "collider3":   ["-M", "3", "--graph", "0->2,1->2"],
                         "confounder3": ["-M", "3", "--graph", "0->1,0->2,1->2"],
                         "chain4":      ["-M", "4", "--graph", "0->1->2->3"],
                         "chain5":      ["-M", "5", "--graph", "0->1->2->3->4"],
                         "chain6":      ["-M", "6", "--graph", "0->1->2->3->4->5"],
                         "chain7":      ["-M", "7", "--graph", "0->1->2->3->4->5->6"],
                         "chain8":      ["-M", "8", "--graph", "0->1->2->3->4->5->6->7"],
                         "full3":       ["-p", "confounder3"], # Equivalent!
                         "full4":       ["-M", "4", "--graph", "0->1->2->3,0->2,0->3,1->3"],
                         "full5":       ["-M", "5", "--graph", "0->1->2->3->4,0->2->4,0->3,0->4,1->3,1->4"],
                         "full6":       ["-M", "6", "--graph", "0->1->2->3->4->5,0->2->4,0->3->5,0->4,0->5,1->3,1->4,1->5,2->5"],
                         "full7":       ["-M", "7", "--graph", "0->1->2->3->4->5->6,0->2->4->6,0->3->5,0->4,0->5,0->6,1->3,1->4,1->5,1->6,2->5,2->6,3->6"],
                         "full8":       ["-M", "8", "--graph", "0->1->2->3->4->5->6->7,0->2->4->6,0->3->5->7,0->4->7,0->5,0->6,0->7,1->3->6,1->4,1->5,1->6,1->7,2->5,2->6,2->7,3->7"]},
                help="Named experiment presets for commonly-used settings.")
            optp = argp.add_argument_group("Optimizers", "Tunables for all optimizers.")
            optp.add_argument("--model-optimizer", "--mopt", action=nauka.ap.Optimizer,
                default="nag:0.001,0.9",
                help="Model Optimizer selection.")
            optp.add_argument("--gamma-optimizer", "--gopt", action=nauka.ap.Optimizer,
                default="nag:0.0001,0.9",
                help="Gamma Optimizer selection.")
            optp.add_argument("--lmaxent",              default=0.000,        type=float,
                help="Regularizer for maximum entropy")
            optp.add_argument("--lsparse",              default=0.000,        type=float,
                help="Regularizer for sparsity.")
            optp.add_argument("--ldag",                 default=0.100,        type=float,
                help="Regularizer for DAGness.")
            dbgp = argp.add_argument_group("Debugging", "Flags for debugging purposes.")
            dbgp.add_argument("--summary",              action="store_true",
                help="Print a summary of the network.")
            dbgp.add_argument("--fastdebug",            action=nauka.ap.FastDebug)
            dbgp.add_argument("--pdb",                  action="store_true",
                help="""Breakpoint before run start.""")
        
        @classmethod
        def run(kls, a):
            from   causal.experiment import Experiment;
            if a.pdb: pdb.set_trace()
            return Experiment(a).rollback().run().exitcode


def main(argv=sys.argv):
    argp = root.addAllArgs()
    try:    import argcomplete; argcomplete.autocomplete(argp)
    except: pass
    a = argp.parse_args(argv[1:])
    a.__argv__ = argv
    return a.__cls__.run(a)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
