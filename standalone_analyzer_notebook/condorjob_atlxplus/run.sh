#!/bin/bash

# Load python environment from work node
source /cvmfs/sft.cern.ch/lcg/views/LCG_96python3/x86_64-centos7-gcc8-opt/setup.sh 

echo ${1}

echo ""
ls -ltrh
echo ""

echo "python3 translate_and_makeTable_etroc2_data.py -d ${1}"
python3 translate_and_makeTable_etroc2_data.py -d ${1}

#echo ""
#ls -ltrh ${1}
#echo ""

# tar output
#echo "Run tar"
#echo "tar -zcvf ./${1}_translated.tgz ${1}/output"
#tar -zcvf ./${1}_translated.tgz ${1}/output 
