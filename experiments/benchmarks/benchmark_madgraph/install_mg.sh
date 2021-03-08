#!/usr/bin/env bash


#This file downloawds MG5 and LHAPDF so that the benchmarking example can be run without an existing installation of MG5
if [ ! -d MG5_aMC_v2_8_3_2 ] 
then
echo "MG5 installation started"
wget https://launchpad.net/mg5amcnlo/2.0/2.8.x/+download/MG5_aMC_v2.8.3.2.tar.gz
tar -xf MG5_aMC_v2.8.3.2.tar.gz
echo "MG5 installation succeded. LHAPDF installation started"
python3.8 MG5_aMC_v2_8_3_2/bin/mg5_aMC --logging=ERROR --file=lhapdf_install.mg5 2>&1> lhapdf_install_log.txt 2>&1
cd MG5_aMC_v2_8_3_2/HEPTools/lhapdf6_py3/share/LHAPDF
wget http://lhapdfsets.web.cern.ch/lhapdfsets/current/NNPDF23_nlo_as_0119.tar.gz
tar -xf NNPDF23_nlo_as_0119.tar.gz
echo "LHAPDF installation succeeded"
fi
