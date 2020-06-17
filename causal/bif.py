# -*- coding: utf-8 -*-
import gzip
import io
import itertools
import os.path
import re
import numpy as np


class BifGraph:
    RE_STRIP_SL_COMMENTS = re.compile('//.*?$', re.M)
    RE_STRIP_ML_COMMENTS = re.compile('/\\*.*?\\*/', re.S)
    RE_TOKENIZE          = re.compile('''
        \s*                                                 # Skip preceding whitespace.
        (                                                   # The set of tokens we may capture, including:
           [\[\]{}();,|]                                 |  # The single control characters []{}();,|
           (?:[a-zA-Z0-9_<=>/\.+-]+)                        # Identifiers and keywords.
        )
    ''', re.S|re.A|re.X)
    
    def __init__(self, path=None):
        if path is None:           self.empty()
        elif os.path.exists(path): self.load(path)
        elif isinstance(path,str): self.loads(path)
        else:                      raise TypeError
    
    @property
    def version(self):
        return int(self.buffer[ 0: 8].view(np.uint64))
    @property
    def num_variables(self):
        return int(self.buffer[ 8:16].view(np.uint64))
    @property
    def max_choices(self):
        return int(self.buffer[16:24].view(np.uint64))
    @property
    def max_indegree(self):
        return int(self.buffer[24:32].view(np.uint64))
    @property
    def num_dofs(self):
        return sum([v.num_dofs for v in self.var_name_dict.values()])
    @property
    def num_params(self):
        return sum([v.cpt.size for v in self.var_name_dict.values()])
    
    def __len__(self):
        return self.num_variables
    
    def __str__(self):
        return self.dumps()
    
    def __repr__(self):
        return 'BifGraph("""\n'+str(self)+'""")'
    
    def __getitem__(self, key):
        """
        All-purpose dict-like interface of graph.
        
        If the key is an integer n, retrieve the n'th variable.
        If the key is a string, retrive the variable with that name.
        If the key is a variable, retrieve its integer index.
        Otherwise die.
        """
        if   isinstance(key, int):
            return self.var_list[key]
        elif isinstance(key, str):
            return self.var_name_dict[key]
        else:
            try:
                return self.var_list.index(key)
            except:
                raise KeyError("Invalid key "+str(key)+" !")
    
    def _peek(self):
        return self._tok[self._idx] if self._idx < len(self._tok) else None
    
    def _consume(self, equals=None, *, type=str):
        v = type(self._tok[self._idx])
        assert(equals is None or v == equals)
        self._idx += 1
        return v
    
    def _parse(self):
        self._parse_network()
        while True:
            if   self._peek() is None:          break
            elif self._peek() == 'variable':    self._parse_variable()
            elif self._peek() == 'probability': self._parse_probability()
            else: raise InvalidValue('Parsing stopped at token '+str(self._peek()))
    
    def _parse_network(self):
        assert(self._ast["name"] is None)
        self._consume('network')
        self._ast["name"] = self._consume(type=str)
        self._consume('{')
        self._consume('}')
    
    def _parse_variable(self):
        self._consume('variable')
        varname = self._require_undeclared(self._consume(type=str))
        self._consume('{')
        self._consume('type')
        self._consume('discrete')
        self._consume('[')
        varnumchoices = self._consume(type=int)
        self._consume(']')
        self._consume('{')
        varchoices = []
        for i in range(varnumchoices):
            if i > 0:
                self._consume(',')
            varchoices.append(self._consume(type=str))
        self._consume('}')
        self._consume(';')
        self._consume('}')
        self._ast["idx"] [varname] = len(self._ast["idx"])
        self._ast["vars"][varname] = BifDiscreteVariableASTNode(varname, varchoices)
    
    def _parse_probability(self):
        self._consume('probability')
        self._consume('(')
        var = self._require_declared(self._consume(type=str))
        while self._peek() != ')':
            if self._peek() in {'|', ','}:
                self._consume()
            else:
                var.ancestors.append(self._require_declared(self._consume(type=str)))
        self._consume(')')
        assert var.cpt is None
        var.make_table()
        self._consume('{')
        while self._peek() != '}':
            self._parse_probability_row(var)
        self._consume('}')
    
    def _parse_probability_row(self, var):
        k = []
        if self._peek() == 'table':
            self._consume('table')
        else:
            self._consume('(')
            for i,v in enumerate(var.ancestors):
                if i>0:
                    self._consume(',')
                k.append(v[self._consume(type=str)])
            self._consume(')')
        k = tuple(k)
        for i in range(var.num_choices):
            if i > 0:
                self._consume(',')
            var.cpt[k][i] = self._consume(type=float)
        self._consume(';')
    
    def _require_declared(self, varname):
        try:
            return self._ast["vars"][varname]
        except:
            raise ValueError('Variable '+varname+' not yet declared!')
    
    def _require_undeclared(self, varname):
        if varname in self._ast["vars"]:
            raise ValueError('Variable '+varname+' already declared!')
        return varname
    
    def _compute_buffer(self):
        # 
        # FORMAT:
        #     [0]    version=0
        #     [1]    self.num_variables
        #     [2]    self.max_choices
        #     [3]    self.max_indegree
        #            for i in range(self.num_variables):
        #     [4]      var[i].num_choices
        #     [5]      var[i].num_ancestors
        #     [6]      var[i].offset_to_ancestors_list
        #     [7]      var[i].offset_to_strides_list
        #     [8]      var[i].offset_to_cpt
        #     ...    
        #            for i in range(self.num_variables):
        #     [A]      var[i].ancestors[:]
        #     [B]      var[i].strides[:]
        #     ...    
        #     [C]    for i in range(self.num_variables):
        #              log(var[i].cpt[:])
        #     ...    
        #     [D]    ZERO-PAD
        #     ...    
        #     [D+63] ZERO-PAD
        #
        
        assert None not in self.var_list
        
        # COMPUTE SIZE IN BYTES
        offset_to_ancestors = []
        offset_to_strides = []
        offset_to_cpt = []
        size  = 4*8
        size += 5*len(self.var_list)*8
        for v in self.var_list:
            offset_to_ancestors.append(size)
            size += v.num_ancestors*8
            offset_to_strides.append(size)
            size += (v.num_ancestors+1)*8
        frontendsize = size
        for v in self.var_list:
            offset_to_cpt.append(size)
            size += v.cpt.size*4
        size += 64*8
        
        # ALLOCATE
        buffer    = np.zeros(size, dtype=np.uint8)
        bufferu64 = buffer[:frontendsize].view(np.uint64)
        bufferf32 = buffer[frontendsize:].view(np.float32)
        
        # FILL
        bufferu64[0] = 0
        bufferu64[1] = len(self.var_list)
        bufferu64[2] = max([v.num_choices   for v in self.var_name_dict.values()], default=0)
        bufferu64[3] = max([v.num_ancestors for v in self.var_name_dict.values()], default=0)
        for i,v in enumerate(self.var_list):
            bufferu64[4+5*i+0] = v.num_choices
            bufferu64[4+5*i+1] = v.num_ancestors
            bufferu64[4+5*i+2] = offset_to_ancestors[i]
            bufferu64[4+5*i+3] = offset_to_strides[i]
            bufferu64[4+5*i+4] = offset_to_cpt[i]
        for i,v in enumerate(self.var_list):
            c  = v.cpt.astype(np.float32, 'C', 'same_kind')
            a  = np.asarray([self[a] for a in v.ancestors], dtype=np.uint64)
            s  = np.asarray(c.strides, dtype=np.uint64)
            
            ba = buffer[
                offset_to_ancestors[i]:
                offset_to_ancestors[i]+a.size*a.itemsize
            ].view(a.dtype).reshape(a.shape)
            bs = buffer[
                offset_to_strides[i]:
                offset_to_strides[i]+s.size*s.itemsize
            ].view(s.dtype).reshape(s.shape)
            bc = buffer[
                offset_to_cpt[i]:
                offset_to_cpt[i]+c.size*c.itemsize
            ].view(c.dtype).reshape(c.shape)
            
            ba[:] = a
            bs[:] = s
            bc[:] = c
            
            v.cpt = bc  # Alias v.cpt onto the raw buffer. Facilitates Python/C interop.
        self.buffer = buffer
    
    def _toposort(self):
        """Topologically sort the variables list, inefficiently."""
        new_list = []
        new_set  = set()
        
        while len(new_list) != len(self.var_list):
            for v in self.var_list:
                if v in new_set:
                    continue
                
                anc_set = set(v.ancestors)
                if new_set.issuperset(anc_set):
                    new_list.append(v)
                    new_set.add(v)
                
                # We have a violation of topological order.
                # We therefore don't add the node to the new_list, and a second
                # pass (and possibly more) will be required.
        
        self.var_list = new_list
    
    def sample(self, out=1):
        """Sample from this graph into a tensor of shape (self.num_variables, batch_size)"""
        if isinstance(out, int):
            out = np.empty((self.num_variables, out), dtype=np.int32)
        
        assert out.itemsize == 4
        assert out.shape[0] == self.num_variables
        for l in range(out.shape[1]):
            for k,v in enumerate(self.var_list):
                p=v.cpt[tuple(out[self[a],l] for a in v.ancestors)]
                out[k,l] = np.random.choice(v.num_choices, p=p)
        return out
    
    def thermalize(self, *args, **kwargs):
        """
        Modify the temperature of the CPTs, either by
          1. An unconditional temperature scaling T or
          2. A  conditional temperature scaling T of any CPT row with a minimum
             non-zero probability less than a threshold Tpthresh
        """
        
        for v in self.var_list:
            v.thermalize(*args, **kwargs)
        return self
    
    def adjacency_matrix(self, dtype=np.float32):
        M = np.zeros((self.num_variables, self.num_variables), dtype=dtype)
        for i,v in enumerate(self.var_list):
            for a in v.ancestors:
                M[i, self[a]] = 1
        return M
    
    def empty(self):
        self.name = None
        self.var_list = []
        self.var_name_dict = {}
        self._compute_buffer()
        return self
    
    def dump(self, fp):
        """Dump BIF to file-like stream"""
        if isinstance(fp, str):
            with open(fp, "w") as fp:
                return self.dump(fp) # Recursive
        
        fp.write("network "+self.name+" {}\n\n")
        for v in self.var_list:
            fp.write(v._get_variable_block())
            fp.write("\n")
        fp.write("\n")
        for v in self.var_list:
            fp.write(v._get_probability_block())
            fp.write("\n")
    
    def dumps(self):
        """Dump BIF to string"""
        s = io.StringIO()
        self.dump(s)
        return s.getvalue()
    
    def load(self, fp):
        """Load BIF from file-like stream"""
        if isinstance(fp, (str, bytes, os.PathLike)):
            try:
                with gzip.open(fp, "rt") as fpgz:
                    return self.load(fpgz) # Recursive
            except:
                with open(fp, "r") as fp:
                    return self.load(fp) # Recursive
        
        return self.loads(fp.read())
    
    def loads(self, s):
        """Load BIF from string"""
        # Empty current state
        self.empty()
        
        # Read
        self._tok = s
        
        # Lex/Tokenize
        self._tok = self.RE_STRIP_SL_COMMENTS.sub(' ', self._tok)
        self._tok = self.RE_STRIP_ML_COMMENTS.sub(' ', self._tok)
        self._tok = tuple(self.RE_TOKENIZE.findall(self._tok))
        self._idx = 0
        
        # Parse
        self._ast = {"name": None,
                     "vars": {},
                     "idx":  {}}
        self._parse()
        if self.__dict__.pop("_idx") != len(self.__dict__.pop("_tok")):
            raise ValueError("Failed to parse BIF file to the end!")
        if self._ast["name"] is None:
            raise ValueError("Network has no name!")
        
        # Semantic Analysis: Translate AST to graph.
        self.name          = self._ast["name"]
        self.var_list      = [None]*len(self._ast["vars"])
        self.var_name_dict = {}
        for name, astvar in self._ast["vars"].items():
            idx  = self._ast["idx"][astvar.name]
            var  = BifDiscreteVariable(astvar.name, tuple(astvar.choices), astvar.cpt)
            self.var_name_dict[name] = var
            self.var_list     [idx]  = var
        for name, astvar in self._ast["vars"].items():
            var = self.var_name_dict[name]
            for ancvar in astvar.ancestors:
                var.ancestors.append(self.var_name_dict[ancvar.name])
            var.ancestors = tuple(var.ancestors)
        self.__dict__.pop("_ast")
        
        # Finalize
        self._toposort()       # Topographically sort the nodes if required.
        self._compute_buffer() # Compact the graph for high-speed reading in C.
        return self
    
    def clone(self):
        return BifGraph().loads(self.dumps())

class BifDiscreteVariableASTNode:
    __slots__ = ['name', 'choices', 'ancestors', 'cpt']
    def __init__(self, name, choices):
        self.name      = name
        self.choices   = choices
        self.ancestors = []
        self.cpt       = None
    
    @property
    def num_choices(self): return len(self.choices)
    @property
    def num_ancestors(self): return len(self.ancestors)
    @property
    def table_shape(self):
        return tuple(a.num_choices for a in self.ancestors) + (self.num_choices,)
    def make_table(self, dtype=np.float32, *args, **kwargs):
        self.cpt = np.empty(self.table_shape, dtype=dtype, *args, **kwargs)
        self.cpt.fill(np.nan)
    def __getitem__(self, k):
        if isinstance(k, str):
            return self.choices.index(k)
        return self.cpt[k] 


class BifVariable:
    def __init__(self, name):
        self.name = name
    
    def __str__(self):
        return self._get_variable_block()
    
    def __repr__(self):
        return str(self)
    
    def _get_variable_block(self):
        raise NotImplementedError
    
    def _get_probability_block(self):
        raise NotImplementedError


class BifDiscreteVariable(BifVariable):
    def __init__(self, name, choices, cpt):
        assert cpt.shape[-1] == len(choices)
        super().__init__(name)
        self.choices   = choices
        self.cpt       = cpt
        self.ancestors = []
    
    def __getitem__(self, key):
        """
        Multi-purpose dict-like interface of variable.
        
        If the key is a string, retrive integer index of the choice with that name.
        Otherwise, index the conditional probability table.
        """
        if isinstance(key, str):
            return self.choices.index(key)
        return self._cpt[key]
    
    @property
    def num_choices(self):
        return len(self.choices)
    @property
    def num_ancestors(self):
        return len(self.ancestors)
    @property
    def num_dofs(self):
        return self.cpt[...,:-1].size  # Because they sum to 1.
    
    def _get_variable_block(self):
        s  = ""
        s += "variable "+self.name+" { "
        s += "type discrete [ "+str(self.num_choices)+" ] { "
        s += ", ".join(self.choices)
        s += " }; }"
        return s
    
    def _get_probability_block(self):
        s  = ""
        s += "probability ( "+self.name+" "
        if self.num_ancestors == 0:
            s += ") {\n"
            s += "  table " + ", ".join(map(str, self.cpt)) + ";\n"
            s += "}"
        else:
            ancname    = [a.name    for a in self.ancestors]
            ancchoices = itertools.product(*[a.choices for a in self.ancestors])
            cptrows    = self.cpt.reshape(-1, self.cpt.shape[-1])
            s += "| " + ", ".join(ancname) + " ) {\n"
            for ancchoice, cptrow in zip(ancchoices, cptrows):
                s += "  (" + ", ".join(ancchoice)
                s += ") "  + ", ".join(map(str, cptrow)) + ";\n"
            s += "}"
        return s
    
    def thermalize(self, T, Tpthresh=1):
        """
        Modify the temperature of the CPTs, either by
          1. An unconditional temperature scaling T or
          2. A  conditional temperature scaling T of any CPT row with a minimum
             non-zero probability less than a threshold Tpthresh
        """
        assert(T > 0)
        assert(Tpthresh >= 0 and Tpthresh <= 1)
        
        if T==1:
            return self
        
        nzmask      = self.cpt != 0
        newcpt      = self.cpt.astype(np.float64)
        newcpt      = np.where(nzmask, newcpt, np.nan)
        fixmask     = np.nanmin(newcpt, axis=-1, keepdims=True) < Tpthresh
        T           = np.where(fixmask, T, 1)
        newcpt      = np.exp(np.log(newcpt)/T)
        newcpt      = np.where(nzmask, newcpt, 0)
        newcpt     /= newcpt.sum(axis=-1, keepdims=True)
        self.cpt[:] = newcpt.astype(self.cpt.dtype)
        return self



def empty():
    """Return empty Graph"""
    return BifGraph()
def load(fp):
    """Deserialize `fp` stream to a BifGraph."""
    return BifGraph().load(fp)
def loads(s):
    """Deserialize `s` string to a BifGraph."""
    return BifGraph().loads(s)
def dump(g, fp):
    """Serialize `g` as a BIF-formatted stream to `fp`."""
    return g.dump(fp)
def dumps(g):
    """Serialize `g` to a BIF-formatted `str`"""
    return g.dumps()

