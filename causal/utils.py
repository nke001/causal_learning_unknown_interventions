import re
import numpy as np



def parse_skeleton(graph, M=None):
    """
    Parse the skeleton of a causal graph in the mini-language of --graph.
    
    The mini-language is:
        
        GRAPH      = ""
                     CHAIN{, CHAIN}*
        CHAIN      = INT_OR_SET {-> INT_OR_SET}
        INT_OR_SET = INT | SET
        INT        = [0-9]*
        SET        = \{ SET_ELEM {, SET_ELEM}* \}
        SET_ELEM   = INT | INT_RANGE
        INT_RANGE  = INT - INT
    """
    
    regex = re.compile(r'''
        \s*                                      # Skip preceding whitespace
        (                                        # The set of tokens we may capture, including
          [,]                                  | # Commas
          (?:\d+)                              | # Integers
          (?:                                    # Integer set:
            \{                                   #   Opening brace...
              \s*                                #   Whitespace...
              \d+\s*(?:-\s*\d+\s*)?              #   First integer (range) in set...
              (?:,\s*\d+\s*(?:-\s*\d+\s*)?\s*)*  #   Subsequent integers (ranges)
            \}                                   #   Closing brace...
          )                                    | # End of integer set.
          (?:->)                                 # Arrows
        )
    ''', re.A|re.X)
    
    # Utilities
    def parse_int(s):
        try:    return int(s.strip())
        except: return None
    
    def parse_intrange(s):
        try:
            sa, sb = map(str.strip, s.strip().split("-", 1))
            sa, sb = int(sa), int(sb)
            sa, sb = min(sa,sb), max(sa,sb)+1
            return range(sa,sb)
        except:
            return None
    
    def parse_intset(s):
        try:
            i = set()
            for s in map(str.strip, s.strip()[1:-1].split(",")):
                if parse_int(s) is not None: i.add(parse_int(s))
                else:                        i.update(set(parse_intrange(s)))
            return sorted(i)
        except:
            return None
    
    def parse_either(s):
        asint = parse_int(s)
        if asint is not None: return asint
        asset = parse_intset(s)
        if asset is not None: return asset
        raise ValueError
    
    def find_max(chains):
        m = 0
        for chain in chains:
            for link in chain:
                link = max(link) if isinstance(link, list) else link
                m = max(link, m)
        return m
    
    # Crack the string into a list of lists of (ints | lists of ints)
    graph  = [graph] if isinstance(graph, str) else graph
    chains = []
    for gstr in graph:
        for chain in re.findall("((?:[^,{]+|\{.*?\})+)+", gstr, re.A):
            links = list(map(str.strip, regex.findall(chain)))
            assert(len(links)&1)
            
            chain = [parse_either(links.pop(0))]
            while links:
                assert links.pop(0) == "->"
                chain.append(parse_either(links.pop(0)))
            chains.append(chain)
    
    # Find the maximum integer referenced within the skeleton
    uM = find_max(chains)+1
    if M is None:
        M = uM
    else:
        assert(M >= uM)
        M = max(M, uM)
    
    # Allocate adjacency matrix.
    gamma = np.zeros((M,M), dtype=np.float32)
    
    # Interpret the skeleton
    for chain in chains:
        for prevlink, nextlink in zip(chain[:-1], chain[1:]):
            if   isinstance(prevlink, list) and isinstance(nextlink, list):
                for i in nextlink:
                    for j in prevlink:
                        if i>j:
                            gamma[i,j] = 1
            elif isinstance(prevlink, list) and isinstance(nextlink, int):
                for j in prevlink:
                    if nextlink>j:
                        gamma[nextlink,j] = 1
            elif isinstance(prevlink, int)  and isinstance(nextlink, list):
                minn = min(nextlink)
                if   minn == prevlink:
                    raise ValueError("Edges are not allowed from " +
                                     str(prevlink) + " to oneself!")
                elif minn <  prevlink:
                    raise ValueError("Edges are not allowed from " +
                                     str(prevlink) + " to ancestor " +
                                     str(minn) + " !")
                else:
                    for i in nextlink:
                        gamma[i,prevlink] = 1
            elif isinstance(prevlink, int)  and isinstance(nextlink, int):
                if   nextlink == prevlink:
                    raise ValueError("Edges are not allowed from " +
                                     str(prevlink) + " to oneself!")
                elif nextlink <  prevlink:
                    raise ValueError("Edges are not allowed from " +
                                     str(prevlink) + " to ancestor " +
                                     str(nextlink) + " !")
                else:
                    gamma[nextlink,prevlink] = 1
    
    # Return adjacency matrix.
    return gamma
