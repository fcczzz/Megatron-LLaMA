#!/bin/bash
mkdir ../../dataset
cp trans.py ../../dataset
cd ../../dataset
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --repo-type dataset --resume-download AhmedSSoliman/CodeSearchNet-py --local-dir CodeSearchNet-py
python trans.py
rm origin.json
#最后得到code-search-net-python.json