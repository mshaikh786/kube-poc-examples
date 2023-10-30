import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Report generator for BERT benchmark.')
parser.add_argument('--file', type=str,
                    help='Filename with absolute path to process')
args = parser.parse_args()


fh = open(args.file, 'r', encoding='utf-8', errors='ignore')

buf = fh.readlines()

fh.close()


if __name__ == '__main__':

    vals=list()

    for line in buf:
        if 'real' in line:
            vals.append(float(line.strip().split()[1]))
    
    print("Min: %f\nMax: %f\nStd dev: %f" %(np.min(vals),np.max(vals),np.std(vals)))    


