#!/usr/bin/env bash

source ./configy.txt
python3 -m pip install -e torchps 
full_path=$(realpath $0)
dir_path=$(dirname $full_path)
python3 $mg/bin/mg5_aMC --logging=ERROR --file=$mgscript 2>&1> mg5_log.txt 2>&1
cd ./$name/SubProcesses
export LD_LIBRARY_PATH=$lhapdfpath
for f in *; 
    do   if [ -d "$f" ]; 
        then cd $f; 
        make matrix2py.so 2>&1> f2py_log.txt 2>&1;
        mv matrix2py* matrix2py.so
        mkdir -p $bench/utils/integrands/mg/$f/param 
        cd ..
        cd ..
        cp ./Cards/param_card.dat $bench/utils/integrands/mg/$f/param 
        cd SubProcesses/$f
        cp {nexternal.inc,pmass.inc,param.log} $bench/utils/integrands/mg/$f/param 
        cp matrix2py.so $bench/utils/integrands/mg/$f
        python3 $bench/benchmark_madgraph.py --process $f --pdf_type $lhapdf --pdf_dir $lhapdfdir --db madgraph_$f --experiment_name madgraph_$f --config $config --debug --lhapdf_dir $lhapdfpath/python3.6/site-packages --no-pdf
        mv madgraph_$f $bench/
        cd ..;
    fi
done

#configy, python3.6, 
#config, python3.7

