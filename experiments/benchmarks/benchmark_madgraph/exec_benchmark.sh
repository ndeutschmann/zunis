#!/usr/bin/env bash


#This is the main execution script. It reads system-specific paths from the config file, installs the phase-space generator, executes MadGraph5_aMC, generates the matrix elements and then performs MC integration for all subprocesses.

source ./config.txt
python3 -m pip install git+https://github.com/NGoetz/TorchPS
full_path=$(realpath $0)
dir_path=$(dirname $full_path)
if [ -d "$name" ]; then rm -Rf ./$name; fi
python3 $mg/bin/mg5_aMC --logging=ERROR --file=$mgscript 2>&1> mg5_log.txt 2>&1
cd ./$name/SubProcesses
export LD_LIBRARY_PATH=$lhapdfdir/lib
for f in *; 
    do   if [ -d "$f" ]; 
        then cd $f; 
        make matrix2py.so 2>&1> f2py_log.txt 2>&1;
        rm -f matrix2py.so
        mv matrix2py* matrix2py.so
        mkdir -p $bench/utils/integrands/mg/$f/param 
        cd ..
        cd ..
        cp ./Cards/param_card.dat $bench/utils/integrands/mg/$f/param 
        cd SubProcesses/$f
        cp {nexternal.inc,pmass.inc,param.log} $bench/utils/integrands/mg/$f/param 
        cp matrix2py.so $bench/utils/integrands/mg/$f
        python3 $bench/benchmark_madgraph.py --process $f --pdf_type $lhapdf --pdf_dir $lhapdfdir/share/LHAPDF --db madgraph_$f --experiment_name madgraph_$f --config $config --debug --lhapdf_dir $lhapdfdir/lib/python3.7/site-packages --pdf
        mv madgraph_$f $bench/
        cd ..;
    fi
done

