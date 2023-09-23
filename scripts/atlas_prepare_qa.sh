#!/usr/bin/env bash

DATA_DIR=/mimer/NOBACKUP/groups/snic2022-22-1003/APP/qa-retriever/data/atlas/

echo "PREPARING DATA [QA] => ${DATA_DIR}"
python preprocessing/atlas/prepare_qa.py --output_directory ${DATA_DIR} 

echo "DUMPING ONE SAMPLE FROM POPQA"
head -n 1 ${DATA_DIR}/popqa_data/test.jsonl | python -c 'import json,sys; print(json.dumps(json.loads(sys.stdin.read()), indent=4))' > ${DATA_DIR}/popqa_data/small.test.jsonl

echo "DUMPING ONE SAMPLE FROM NQ"
head -n 1 ${DATA_DIR}/nq_data/train.jsonl | python -c 'import json,sys; print(json.dumps(json.loads(sys.stdin.read()), indent=4))' > ${DATA_DIR}/nq_data/small.train.jsonl

echo "DUMPING ONE SAMPLE FROM TRIVIAQA"
head -n 1 ${DATA_DIR}/triviaqa_data/train.jsonl | python -c 'import json,sys; print(json.dumps(json.loads(sys.stdin.read()), indent=4))' > ${DATA_DIR}/triviaqa_data/small.train.jsonl