import os,sys,re,torch


def main(argv):
    allgammagt = []
    allgammalr = []
    
    # Iterate over every file at first.
    for a in argv[1:]:
        with open(a, "r") as f:
            readinggt = False
            readinglr = False
            gammagt   = []
            gammalr   = []
            for line in f:
                if "slurmstepd:" in line:
                    break
                
                if   readinggt:
                    stringgt += line
                    readinggt = ")" not in line
                    if not readinggt:
                        gammagt.append(eval("torch."+stringgt))
                elif readinglr:
                    stringlr += line
                    readinglr = ")" not in line
                    if not readinglr:
                        gammalr.append(eval("torch."+stringlr))
                
                if   "] Gamma GT:" in line:
                    stringgt  = ""
                    readinggt = True
                elif "] Gamma:"    in line:
                    stringlr  = ""
                    readinglr = True
        
        allgammagt.append(gammagt)
        allgammalr.append(gammalr)
    
    # Having collected the series, select the last one and compute its ROC.
    lastidx    = min([len(l) for l in allgammagt]+[len(l) for l in allgammalr])-1
    allgammagt = torch.stack([l[lastidx] for l in allgammagt])
    allgammalr = torch.stack([l[lastidx] for l in allgammalr])
    fpr, tpr = roc(allgammagt, allgammalr)
    for fp,tp in zip(fpr, tpr):
        print("{:f},{:f}".format(float(fp), float(tp)))
    #print(float(auc(allgammagt, allgammalr)))


def eq0nodiag(x):
    allz   = x.eq(0).long().sum((1,2,3))
    diagz  = x.diagonal(0,2,3).eq(0).long().sum((1,2))
    return allz-diagz


def ne0nodiag(x):
    allnz  = x.ne(0).long().sum((1,2,3))
    diagnz = x.diagonal(0,2,3).ne(0).long().sum((1,2))
    return allnz-diagnz


def roc(gammagts, gammalrs, thresh=torch.linspace(1,0,1001)):
    thresh   = thresh.view(-1,1,1,1)
    gammalrt = gammalrs.unsqueeze(0).ge(thresh)
    gammagts = gammagts.unsqueeze(0)
    gammagtn = gammagts.ne(0)
    gammagtz = gammagts.eq(0)
    cp       = ne0nodiag(gammagts).float()
    cn       = eq0nodiag(gammagts).float()
    tp       = ne0nodiag(gammagtn.mul(gammalrt)).float()
    fp       = ne0nodiag(gammagtz.mul(gammalrt)).float()
    fpr      = torch.where(cn.eq(0), torch.zeros_like(fp), fp/cn)
    tpr      = torch.where(cp.eq(0), torch.zeros_like(tp), tp/cp)
    return fpr, tpr


def auc(gammagts, gammalrs):
    x,y = roc(gammagts, gammalrs)
    dx = x[1:]-x[:-1]
    ay = 0.5*(y[1:]+y[:-1])
    return float(dx.mul(ay).sum())


if __name__=="__main__":
    main(sys.argv)
