#
# Erases interruptions.
#
import os,sys,pdb,re


def main(argv):
    # Allow reading standard input
    while len(argv) < 2:
        argv.append("/dev/stdin")
    
    #
    # Carve the text logs to remove SLURM interruptions.
    # The strategy we adopt is to select text in whole blocks from either of
    # the regexes '] Command:' or 'Epoch \d+ done.' to the next
    # 'Epoch \d+ done.', unless 'slurmstepd' occurs in-between, in which case
    # selection begins anew from the next '] Command:' after.
    #
    EPOCH_REGEX = re.compile(r"Epoch \d+ done\.")
    with open(argv[1], "r") as f:
        hunting_for_start = True
        epoch_lines       = []
        for line in f:
            line_interrupted = "slurmstepd:" in line
            
            if hunting_for_start:
                if "] Command:" in line and not line_interrupted:
                    hunting_for_start = False
                    epoch_lines = [line]
            else:
                if line_interrupted:
                    hunting_for_start = True
                    epoch_lines = []
                    continue
                
                epoch_lines.append(line)
                
                if EPOCH_REGEX.search(line):
                    for saved_line in epoch_lines:
                        sys.stdout.write(saved_line)
                    sys.stdout.flush()
                    epoch_lines = []


if __name__=="__main__":
    main(sys.argv)
