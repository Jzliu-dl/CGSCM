#!/bin/bash
version=$1
# 
python eval/test.py data/qa-hme $version test 480000 False

