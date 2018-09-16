

import os,subprocess, re

_path = os.path.dirname(os.path.realpath(__file__))

def exact_prime_implicants(minterms):
    if len(minterms) == 0:
        return []
    k = len(minterms[0])
    input_str = ".i %d\n.o 1\n" % k
    input_str = input_str+'\n'.join([i.replace('2','-')+' 1' for i in minterms]) + '\n.e \n'
    p = subprocess.Popen([os.path.join(_path,'espresso'),'-Dprimes'], stdin=subprocess.PIPE,stdout=subprocess.PIPE)
    outs, errs = p.communicate(input_str)
    return [pi.replace('-','2') for pi in re.findall(r'^[-01]+',outs,re.MULTILINE)]
