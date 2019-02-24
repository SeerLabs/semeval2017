"""
Conversion script expecting data in Semeval2017 format
for keyphrase extraction (grouped as txts and ann files)

Returns tokenized terms with BIO notation to identify
keyphrases and offset boundaries.

"""

import os


#Hardcoded paths to change when testing on own input data
input_root ='./semevaltrainingdata'
anns_folder = 'anns'
txt_folder = 'txts'

#Lines will be added in append mode, so if output_fold is non-empty,
# make sure to remove everything in it first

output_fold ='malletformatfeaturesBIO'


txts = sorted( [os.path.join(input_root, txt_folder, txt) for txt in os.listdir(os.path.join(input_root, txt_folder))])
anns = sorted([os.path.join(input_root, anns_folder, ann) for ann in os.listdir(os.path.join(input_root, anns_folder))])
#Kept same naming convention as original experiments
outs = sorted( [os.path.join(output_fold, txt[:-4]+'__output.txt') for txt in os.listdir(os.path.join(input_root, txt_folder))])


for txt, ann, out in zip(txts, anns, outs):

    current_off = 0
    kpw_starts={}
    kpw_offs={}

    with open(txt) as txtf, open(ann) as annf, open(out, 'a') as outf:

        tokens = txtf.read().split()
        # Takes the actual keyphrases from the ann file and offsets
        ann_lines =annf.readlines()

        for line in ann_lines:

            #Skip semantic relations, if any
            if line[0]=='R' or line[0]=="*":
                continue

            offs = line.split('\t')[1].split()
            start = offs[1]

            if len(offs)== 3:
                #Most of cases
                end = line.split('\t')[1].split()[2]

            # Some of semeval annotations have intermediate/nested offsets, but here we only
            # consider start and outermost end
            elif len(offs)==4:

                end = line.split('\t')[1].split()[3]

            kp = line.split('\t')[2][:-1]


            kpw_starts[int(start)] = dict(end=end, kp=kp)
            kpw_offs[(int(start), int(end))] = kp


        """
        OutFormat:
        one token per line
        one empty newline in between sentences
        token <tab> (BIO)-KP <space> start_offset <space> end_offset <\n>
        
        """

        for token in tokens:

            curr_end = current_off + len(token)

            if current_off in kpw_starts:

                bio_tag='B-KP'

            # +1 and +2 tolerance on start offset
            #NOTE" this is to deal with semeval2017 buggy annotations
            #Uncomment the following two elif blocks if other annotated dataset is used
            elif current_off+1 in kpw_starts:

                bio_tag='B-KP'
                current_off = current_off+1
                curr_end = current_off + len(token)

            elif current_off+2 in kpw_starts and len(token)>2:

                bio_tag='B-KP'
                current_off = current_off+2
                curr_end = current_off + len(token)

            else:

                inner = False

                for start,end in kpw_offs.keys():


                    if (current_off > start)&(curr_end <= end+1):

                        inner = True
                        break

                if inner:

                    bio_tag = 'I-KP'

                else:

                    bio_tag = 'O'

            outf.write(token+'\t'+bio_tag+' '+str(current_off)+' '+str(curr_end)+'\n')

            if token[-1]=='.':
                outf.write('\n')

            current_off = curr_end +1

