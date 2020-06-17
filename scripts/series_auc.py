import os,sys,re,torch


def main(argv):
    allgammagt = []
    allgammalr = []
    
    # Iterate over every file at first.
    for a in argv[1:]:
        with open(a, "r") as f:
            readinggt = False
            readinglr = False
            gammalr   = []
            gammagt   = []
            gamma     = None
            for line in f:
                if "slurmstepd:" in line:
                    break
                
                if   readinggt:
                    stringgt.append(line)
                    readinggt = not ("False negative" in line)
                    if not readinggt:
                        stringgt = list(map(lambda x:x[8:-1].split('|')[0], stringgt[1:-1]))
                        stringgt = [s[1::2] for s in stringgt]
                        stringgt = [[int(x in "*#") for x in s] for s in stringgt]
                        gamma    = torch.as_tensor(stringgt)
                        gammagt.append(gamma)
                elif readinglr:
                    stringlr += line
                    readinglr = ")" not in line
                    if not readinglr:
                        gammalr.append(eval("torch."+stringlr))
                
                if   "] Gamma GT:" in line:
                    if gamma is None:
                        stringgt  = []
                        readinggt = True
                    else:
                        gammagt.append(gamma)
                elif "] Gamma:"    in line:
                    stringlr  = ""
                    readinglr = True
        
        allgammagt.append(gammagt)
        allgammalr.append(gammalr)
    
    # Having collected the series, iterate over them and compute the AUC.
    allgammagt = [torch.stack(l) for l in allgammagt]
    allgammalr = [torch.stack(l) for l in allgammalr]
    minlen     = min([len(t) for t in allgammagt]+[len(t) for t in allgammalr])
    allgammagt = torch.stack([l[:minlen] for l in allgammagt], dim=1)
    allgammalr = torch.stack([l[:minlen] for l in allgammalr], dim=1)
    for gammagts,gammalrs in zip(allgammagt, allgammalr):
        print(float(auc(gammagts, gammalrs)))


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
    torch.set_num_threads(1)
    with torch.no_grad():
        main(sys.argv)
