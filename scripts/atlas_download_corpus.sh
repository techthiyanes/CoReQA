#!/usr/bin/env bash

DATA_DIR=/mimer/NOBACKUP/groups/snic2022-22-1003/APP/qa-retriever/data/atlas/
YEAR=${1:-"dec2018"}
CORPUS=corpora/wiki/enwiki-${YEAR}

echo "DOWNLODING ${CORPUS} => ${DATA_DIR}"

python preprocessing/atlas/download_corpus.py --corpus ${CORPUS} --output_directory ${DATA_DIR} 
