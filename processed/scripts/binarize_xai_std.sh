#!/bin/bash
#

[[ -z "$1" ]] && { echo "PREFIX is empty" ; exit 1; }
PREFIX=$1
i=0
src=${PREFIX}_data_raw_stdev/$i
dest=${PREFIX}_data_bin_stdev/$i
[[ -d "${PREFIX}_data_bin_stdev" ]] && { echo "output directory ${PREFIX}_data_bin_stdev already exists" ; exit 1; }

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