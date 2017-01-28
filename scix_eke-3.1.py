#!/usr/bin/python3
# basically the same as scix_eke-3.0.py except that "[" and "]" are tagged as "X", 
# instead of "NN" 
# Compared with EKE2.0, I replace the NLTK POS tagger with the Stanford's 
# run python3 scix_eke-3.0.py -h to see the usage
# make sure to change the output dir if you want to preserve the previous results
import argparse
import logging
import glob
import os
import re
import io
from gen_keyphrase_core_stanford import gen_keyphrases
import nltk
from nltk.tag.stanford import StanfordPOSTagger

nltk.internals.config_java(options='-xmx4G')

def extract_keyph(fpath):
    logger = logging.getLogger("extract_keyph")
    keyphs = []
    contents = io.open(fpath,encoding="utf-8").read()
    return gen_keyphrases(contents)
    #for kp in gen_keyphrases(contents,1):
    #for term,ctr in gen_keyphrases(contents,1).items(): 
    #    keyphs.append((term,[0,1]))
    #return keyphs

def main(confs):
    logger = logging.getLogger("main")

    testdir = confs['testdir']
    outdir = confs['outdir']
    n_processed = 0
    
    # loop over all .txt files under testdir
    for f in glob.glob(os.path.join(testdir,"*.txt")):
        logger.info("processing file: %(1)s"%{"1":f})
        keyphs = extract_keyph(f)

        outfile = re.sub(r"txt$","ann",f).replace(testdir,outdir)
        outf = open(outfile,"w")
        # keyphs is a list of tuples ("keyphrase",[boundarys])
        for i,keyph in enumerate(keyphs):
            line = "T%(0)d\t%(1)s %(2)d %(3)d\t%(4)s\n"%{"0":i,"1":"KEYPHRASE-NOTYPES","2":keyph[1][0],"3":keyph[1][1],"4":keyph[0]}
            outf.write(line)
        outf.close()
        n_processed += 1

    logger.info("processed: %(0)d files output to %(1)s"%{"0":n_processed,"1":outdir})
    logger.info("end")

if __name__ == "__main__":
    # accept input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("testdir",metavar='TESTDIR',type=str,help="A directory containing .txt files to be tested")
    parser.add_argument("outdir",metavar='OUTDIR',type=str,help="A directory containing the output .ann files")
    parser.add_argument("-v","--verbose",help="increase output verbosity",action="store_true",dest="verbose",default=False)
    args = parser.parse_args()
    if args.verbose:
        logging_level = logging.DEBUG
    else:
        logging_level = logging.INFO

    # logging configurations
    logging.basicConfig(level=logging_level,\
                  filename="logs/scix_eke-2.0.log",\
                  format="%(asctime)s %(name)-10s %(levelname)-8s %(message)s",
                  filemode="w")
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(name)-10s %(levelname)-8s %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)
    logging.info("Logging configuration done")

    # pass command line configuration to main program
    confs = {"testdir":args.testdir,"outdir":args.outdir}
    main(confs)
