#!/usr/bin/python3
import argparse
import logging
import glob
import os
import re
import io
from gen_keyphrase_core import gen_term_ctr

def extract_keyph(fpath):
    logger = logging.getLogger("extract_keyph")
    keyphs = []
    contents = io.open(fpath,encoding="utf-8").read()
    for term,ctr in gen_term_ctr(contents,1).items(): 
        keyphs.append((term,[0,1]))
    return keyphs

def main(confs):
    logger = logging.getLogger("main")

    testdir = confs['testdir']
    outdir = confs['outdir']
    n_processed = 0
    
    # loop over all .txt files under testdir
    files = glob.glob(os.path.join(testdir,'*.txt'))
    for f in files:
        logger.info("processing file: %(1)s"%{"1":f})
        keyphs = extract_keyph(f)

        outfile = re.sub(r"txt$","ann",f).replace(testdir,outdir)
        outf = open(outfile,"w")
        # keyphs is a list of tuples ("keyphrase",[boundarys])
        for keyph in keyphs:
            line = "T0\t%(1)s %(2)d %(3)d\t%(4)s\n"%{"1":"PROCESS","2":keyph[1][0],"3":keyph[1][1],"4":keyph[0]}
            outf.write(line)
        outf.close()
        n_processed += 1

    logger.info("%(1)d out of %(2)d files processed"%{"1":n_processed,"2":len(files)})
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
                  filename="logs/scix_eke.log",\
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
