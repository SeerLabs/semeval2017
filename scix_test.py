#!/usr/bin/python3
import argparse
import logging
import glob
import os
import re
import io
from gen_keyphrase_core_bounds import gen_keyphrases

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

    testfile = confs['testfile']
    #outdir = confs['outdir']
    n_processed = 0
    
    # loop over all .txt files under testdir
    #files = glob.glob(os.path.join(testdir,'*.txt'))
    for f in [testfile]:
        logger.info("processing file: %(1)s"%{"1":f})
        keyphs = extract_keyph(f)

        outfile = re.sub(r"txt$","ann",f)
        outf = open(outfile,"w")
        # keyphs is a list of tuples ("keyphrase",[boundarys])
        for i,keyph in enumerate(keyphs):
            line = "T%(0)d\t%(1)s %(2)d %(3)d\t%(4)s\n"%{"0":i,"1":"KEYPHRASE","2":keyph[1][0],"3":keyph[1][1],"4":keyph[0]}
            outf.write(line)
        outf.close()
        n_processed += 1

    logger.info("outfile: %(1)s"%{"1":outfile})
    logger.info("end")

if __name__ == "__main__":
    # accept input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("testfile",metavar='TESTFILE',type=str,help="A .txt file to be tested")
    #parser.add_argument("testdir",metavar='TESTDIR',type=str,help="A directory containing .txt files to be tested")
    #parser.add_argument("outdir",metavar='OUTDIR',type=str,help="A directory containing the output .ann files")
    parser.add_argument("-v","--verbose",help="increase output verbosity",action="store_true",dest="verbose",default=False)
    args = parser.parse_args()
    if args.verbose:
        logging_level = logging.DEBUG
    else:
        logging_level = logging.INFO

    # logging configurations
    logging.basicConfig(level=logging_level,\
                  filename="logs/test.log",\
                  format="%(asctime)s %(name)-10s %(levelname)-8s %(message)s",
                  filemode="w")
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(name)-10s %(levelname)-8s %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)
    logging.info("Logging configuration done")

    # pass command line configuration to main program
    confs = {"testfile":args.testfile}
    main(confs)
