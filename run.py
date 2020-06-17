#!/usr/bin/env python
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
                choices=["cat"],
                help="Model Selection.")
            argp.add_argument("-e", "--num-epochs",     default=200,          type=int,
                help="Number of epochs")
            argp.add_argument("--batch-size", "--bs",   default=256,          type=int,
                help="Batch Size")
            argp.add_argument("--dpe",                  default=10,           type=int,
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
            argp.add_argument("--temperature-gt",       default=1.0,          type=float,
                help="Temperature of the ground-truth graph. Temperatures > 1 lead to more uniform sampling, temperatures < 1 lead to less uniform sampling.")
            argp.add_argument("--temperature-pthresh",  default=1.0,          type=float,
                help="Threshold for temperature adjustment. If a ground-truth CPT row contains a (non-zero) probability lower than this, it is thermalized with --temperature-gt.")
            argp.add_argument("-v", "--verbose",        default=0,            type=int,
                nargs="?",   const=10,
                help="Printing interval")
            argp.add_argument("--cuda",                 action=nauka.ap.CudaDevice)
            argp.add_argument("--graph",                action="append",      type=str,
                default=None,
                help="Graph string.")
            argp.add_argument("-p", "--preset",         action=nauka.ap.Preset,
                choices={"blank3":      ["-M", "3",  "--graph", ""],
                         "chain3":      ["-M", "3",  "--graph", "0->1->2"],
                         "fork3":       ["-M", "3",  "--graph", "0->{1-2}"],
                         "collider3":   ["-M", "3",  "--graph", "{0-1}->2"],
                         "collider4":   ["-M", "4",  "--graph", "{0-2}->3"],
                         "collider5":   ["-M", "5",  "--graph", "{0-3}->4"],
                         "collider6":   ["-M", "6",  "--graph", "{0-4}->5"],
                         "collider7":   ["-M", "7",  "--graph", "{0-5}->6"],
                         "collider8":   ["-M", "8",  "--graph", "{0-6}->7"],
                         "collider9":   ["-M", "9",  "--graph", "{0-7}->8"],
                         "collider10":  ["-M", "10", "--graph", "{0-8}->9"],
                         "collider11":  ["-M", "11", "--graph", "{0-9}->10"],
                         "collider12":  ["-M", "12", "--graph", "{0-10}->11"],
                         "collider13":  ["-M", "13", "--graph", "{0-11}->12"],
                         "collider14":  ["-M", "14", "--graph", "{0-12}->13"],
                         "collider15":  ["-M", "15", "--graph", "{0-13}->14"],
                         "confounder3": ["-M", "3",  "--graph", "{0-2}->{0-2}"],
                         "chain4":      ["-M", "4",  "--graph", "0->1->2->3"],
                         "chain5":      ["-M", "5",  "--graph", "0->1->2->3->4"],
                         "chain6":      ["-M", "6",  "--graph", "0->1->2->3->4->5"],
                         "chain7":      ["-M", "7",  "--graph", "0->1->2->3->4->5->6"],
                         "chain8":      ["-M", "8",  "--graph", "0->1->2->3->4->5->6->7"],
                         "chain9":      ["-M", "9",  "--graph", "0->1->2->3->4->5->6->7->8"],
                         "chain10":     ["-M", "10", "--graph", "0->1->2->3->4->5->6->7->8->9"],
                         "chain11":     ["-M", "11", "--graph", "0->1->2->3->4->5->6->7->8->9->10"],
                         "chain12":     ["-M", "12", "--graph", "0->1->2->3->4->5->6->7->8->9->10->11"],
                         "chain13":     ["-M", "13", "--graph", "0->1->2->3->4->5->6->7->8->9->10->11->12"],
                         "chain14":     ["-M", "14", "--graph", "0->1->2->3->4->5->6->7->8->9->10->11->12->13"],
                         "chain15":     ["-M", "15", "--graph", "0->1->2->3->4->5->6->7->8->9->10->11->12->13->14"],
                         "full3":       ["-p", "confounder3"], # Equivalent!
                         "full4":       ["-M", "4",  "--graph", "{0-3}->{0-3}"],
                         "full5":       ["-M", "5",  "--graph", "{0-4}->{0-4}"],
                         "full6":       ["-M", "6",  "--graph", "{0-5}->{0-5}"],
                         "full7":       ["-M", "7",  "--graph", "{0-6}->{0-6}"],
                         "full8":       ["-M", "8",  "--graph", "{0-7}->{0-7}"],
                         "full9":       ["-M", "9",  "--graph", "{0-8}->{0-8}"],
                         "full10":      ["-M", "10", "--graph", "{0-9}->{0-9}"],
                         "full11":      ["-M", "11", "--graph", "{0-10}->{0-10}"],
                         "full12":      ["-M", "12", "--graph", "{0-11}->{0-11}"],
                         "full13":      ["-M", "13", "--graph", "{0-12}->{0-12}"],
                         "full14":      ["-M", "14", "--graph", "{0-13}->{0-13}"],
                         "full15":      ["-M", "15", "--graph", "{0-14}->{0-14}"],
                         "tree9":       ["-M", "9",  "--graph", "0->1->3->7,0->2->6,1->4,3->8,2->5"],
                         "tree10":      ["-M", "10", "--graph", "0->1->3->7,0->2->6,1->4->9,3->8,2->5"],
                         "tree11":      ["-M", "11", "--graph", "0->1->3->7,0->2->6,1->4->10,3->8,4->9,2->5"],
                         "tree12":      ["-M", "12", "--graph", "0->1->3->7,0->2->6,1->4->10,3->8,4->9,2->5->11"],
                         "tree13":      ["-M", "13", "--graph", "0->1->3->7,0->2->6,1->4->10,3->8,4->9,2->5->11,5->12"],
                         "tree14":      ["-M", "14", "--graph", "0->1->3->7,0->2->6,1->4->10,3->8,4->9,2->5->11,5->12,6->13"],
                         "tree15":      ["-M", "15", "--graph", "0->1->3->7,0->2->6->14,1->4->10,3->8,4->9,2->5->11,5->12,6->13"],
                         "jungle3":     ["-p", "fork3"], # Equivalent!
                         "jungle4":     ["-M", "4",  "--graph", "0->1->3,0->2,0->3"],
                         "jungle5":     ["-M", "5",  "--graph", "0->1->3,1->4,0->2,0->3,0->4"],
                         "jungle6":     ["-M", "6",  "--graph", "0->1->3,1->4,0->2->5,0->3,0->4,0->5"],
                         "jungle7":     ["-M", "7",  "--graph", "0->1->3,1->4,0->2->5,2->6,0->3,0->4,0->5,0->6"],
                         "jungle8":     ["-M", "8",  "--graph", "0->1->3->7,1->4,0->2->5,2->6,0->3,0->4,0->5,0->6,1->7"],
                         "jungle9":     ["-M", "9",  "--graph", "0->1->3->7,3->8,1->4,0->2->5,2->6,0->3,0->4,0->5,0->6,1->7,1->8"],
                         "jungle10":    ["-M", "10", "--graph", "0->1->3->7,3->8,1->4->9,0->2->5,2->6,0->3,0->4,0->5,0->6,1->7,1->8,1->9"],
                         "jungle11":    ["-M", "11", "--graph", "0->1->3->7,3->8,1->4->9,4->10,0->2->5,2->6,0->3,0->4,0->5,0->6,1->7,1->8,1->9,1->10"],
                         "jungle12":    ["-M", "12", "--graph", "0->1->3->7,3->8,1->4->9,4->10,0->2->5->11,2->6,0->3,0->4,0->5,0->6,1->7,1->8,1->9,1->10,2->11"],
                         "jungle13":    ["-M", "13", "--graph", "0->1->3->7,3->8,1->4->9,4->10,0->2->5->11,5->12,2->6,0->3,0->4,0->5,0->6,1->7,1->8,1->9,1->10,2->11,2->12"],
                         "jungle14":    ["-M", "14", "--graph", "0->1->3->7,3->8,1->4->9,4->10,0->2->5->11,5->12,2->6->13,0->3,0->4,0->5,0->6,1->7,1->8,1->9,1->10,2->11,2->12,2->13"],
                         "jungle15":    ["-M", "15", "--graph", "0->1->3->7,3->8,1->4->9,4->10,0->2->5->11,5->12,2->6->13,6->14,0->3,0->4,0->5,0->6,1->7,1->8,1->9,1->10,2->11,2->12,2->13,2->14"],
                         "bidiag3":     ["-p", "confounder3"], # Equivalent!
                         "bidiag4":     ["-M", "4",  "--graph", "{0-1}->{1-2}->{2-3}"],
                         "bidiag5":     ["-M", "5",  "--graph", "{0-1}->{1-2}->{2-3}->{3-4}"],
                         "bidiag6":     ["-M", "6",  "--graph", "{0-1}->{1-2}->{2-3}->{3-4}->{4-5}"],
                         "bidiag7":     ["-M", "7",  "--graph", "{0-1}->{1-2}->{2-3}->{3-4}->{4-5}->{5-6}"],
                         "bidiag8":     ["-M", "8",  "--graph", "{0-1}->{1-2}->{2-3}->{3-4}->{4-5}->{5-6}->{6-7}"],
                         "bidiag9":     ["-M", "9",  "--graph", "{0-1}->{1-2}->{2-3}->{3-4}->{4-5}->{5-6}->{6-7}->{7-8}"],
                         "bidiag10":    ["-M", "10", "--graph", "{0-1}->{1-2}->{2-3}->{3-4}->{4-5}->{5-6}->{6-7}->{7-8}->{8-9}"],
                         "bidiag11":    ["-M", "11", "--graph", "{0-1}->{1-2}->{2-3}->{3-4}->{4-5}->{5-6}->{6-7}->{7-8}->{8-9}->{9-10}"],
                         "bidiag12":    ["-M", "12", "--graph", "{0-1}->{1-2}->{2-3}->{3-4}->{4-5}->{5-6}->{6-7}->{7-8}->{8-9}->{9-10}->{10-11}"],
                         "bidiag13":    ["-M", "13", "--graph", "{0-1}->{1-2}->{2-3}->{3-4}->{4-5}->{5-6}->{6-7}->{7-8}->{8-9}->{9-10}->{10-11}->{11-12}"],
                         "bidiag14":    ["-M", "14", "--graph", "{0-1}->{1-2}->{2-3}->{3-4}->{4-5}->{5-6}->{6-7}->{7-8}->{8-9}->{9-10}->{10-11}->{11-12}->{12-13}"],
                         "bidiag15":    ["-M", "15", "--graph", "{0-1}->{1-2}->{2-3}->{3-4}->{4-5}->{5-6}->{6-7}->{7-8}->{8-9}->{9-10}->{10-11}->{11-12}->{12-13}->{13-14}"],
                },
                help="Named experiment presets for commonly-used settings.")
            optp = argp.add_argument_group("Optimizers", "Tunables for all optimizers.")
            optp.add_argument("--model-optimizer", "--mopt", action=nauka.ap.Optimizer,
                default="nag:0.001,0.9",
                help="Model Optimizer selection.")
            optp.add_argument("--gamma-optimizer", "--gopt", action=nauka.ap.Optimizer,
                default="nag:0.0001,0.9",
                help="Gamma Optimizer selection.")
            optp.add_argument("--lsparse",                   action=nauka.ap.LRSchedule,
                default=0,
                help="Regularizer for sparsity.")
            optp.add_argument("--lmaxent",              default=0.000,        type=float,
                help="Regularizer for maximum entropy")
            optp.add_argument("--ldag",                 default=0.100,        type=float,
                help="Regularizer for DAGness.")
            limp = argp.add_argument_group("Sampling Limits", "Limits for sampling procedures.")
            limp.add_argument("--limit-interventions",  default=0,            type=int,
                help="Maximum number of interventions to perform per variable (default=0=unlimited).")
            limp.add_argument("--limit-samples",        default=0,            type=int,
                help="Maximum number of samples to draw per intervention (default=0=unlimited).")
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
    
    
    class bdagl(nauka.ap.Subcommand):
        class dump(nauka.ap.Subcommand):
            @classmethod
            def addArgs(kls, argp):
                argp.add_argument("--dumpDir",           default="dump")
                argp.add_argument("--num-samples",       default=2560,         type=int,
                    help="Number of samples of the graph.")
                argp.add_argument("--num-interventions", default=1111,      type=int,
                    help="Number of interventions.")
                root.train.addArgs(argp)
            
            @classmethod
            def run(kls, a):
                from   causal.bdagl import Experiment;
                if a.pdb: pdb.set_trace()
                return Experiment(a).run()



def main(argv=sys.argv):
    argp = root.addAllArgs()
    try:    import argcomplete; argcomplete.autocomplete(argp)
    except: pass
    a = argp.parse_args(argv[1:])
    a.__argv__ = argv
    return a.__cls__.run(a)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
