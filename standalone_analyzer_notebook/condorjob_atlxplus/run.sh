#!/bin/bash

echo "tar -xvf ./${1}.tar"
tar -xvf ./${1}.tar
echo ""
ls -ltrh
echo ""

echo "python3 simplified_standalone_translate_etroc2_data.py -d ${1}"
python3 simplified_standalone_translate_etroc2_data.py -d ${1}

echo ""
echo -ltrh ${1}/output
echo ""

# tar output
echo "Run tar"
echo "tar -zcvf ./${1}_translated.tgz ${1}/output"
tar -zcvf ./${1}_translated.tgz ${1}/output 
