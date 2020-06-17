#
# Does approximately grep -e "Gamma CE:" | cut -c 52- | tr -d " "
#
import os,sys,re


def main(argv):
    # Allow reading standard input
    while len(argv) < 2:
        argv.append("/dev/stdin")
    
    # Carve the text logs for Gamma CE printouts
    with open(argv[1], "r") as f:
        for line in f:
            if "slurmstepd:" in line:
                break
            
            if "] Gamma CE:" in line:
                line = line[52:].strip()
                print(line)


if __name__=="__main__":
    main(sys.argv)
