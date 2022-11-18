#!/bin/bash

[[ -z "$1" ]] && { echo "PREFIX is empty" ; exit 1; }
PREFIX=$1
i=0
src=subset_8_16/${PREFIX}_data_raw_apex_8bars/$i
dest=subset_8_16/${PREFIX}_data_bin_apex_8bars/$i
[[ -d "subset_8_16/${PREFIX}_data_bin_apex_8bars" ]] && { echo "subset_8_16/output directory ${PREFIX}_data_bin_apex_8bars already exists" ; exit 1; }

echo "Processing fold $i"
	mkdir -p ${dest}
	fairseq-preprocess \
	    --only-source \
	    --trainpref ${src}/train.txt \
	    --validpref ${src}/test.txt \
	    --destdir ${dest}/input0 \
	    --srcdict ${src}/dict.txt \
	    --workers 24
	fairseq-preprocess \
	    --only-source \
	    --trainpref ${src}/train.label \
	    --validpref ${src}/test.label \
	    --destdir ${dest}/label \
	    --workers 24
	cp ${src}/train.label ${dest}/label/train.label
	cp ${src}/test.label ${dest}/label/valid.label
echo "Done"
