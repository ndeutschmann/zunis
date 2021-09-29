#!/usr/bin/env bash


#This is the main execution script. It installs MG5 and LHAPDF if not present, executes MadGraph5_aMC,
#generates the matrix elements and then performs MC integration for all subprocesses.


full_path=$(realpath $0)
dir_path=$(dirname $full_path)
benchmark_path=$(dirname $dir_path)
./install_mg.sh
export LD_LIBRARY_PATH=$dir_path/MG5_aMC_v2_8_3_2/HEPTools/lhapdf6_py3/lib

if [ -d "fortran_output_ucg" ]; then rm -Rf ./fortran_output_ucg; fi
python3 MG5_aMC_v2_8_3_2/bin/mg5_aMC --logging=ERROR --file=nis_script_uc_ucg.mg5 2>&1> mg5_log3.txt 2>&1
cd ./fortran_output_ucg/SubProcesses
for f in *;
    do   if [ -d "$f" ];
        then cd $f;
        make matrix2py.so 2>&1> f2py_log.txt 2>&1;
        rm -f matrix2py.so
        mv matrix2py* matrix2py.so
        mkdir -p $benchmark_path/utils/integrands/mg/$f/param
        cp $dir_path/fortran_output_ucg/Cards/param_card.dat $benchmark_path/utils/integrands/mg/$f/param
        cp {nexternal.inc,pmass.inc,param.log} $benchmark_path/utils/integrands/mg/$f/param
        cp matrix2py.so $benchmark_path/utils/integrands/mg/$f
        python3 $benchmark_path/benchmark_madgraph.py --process $f --pdf_type NNPDF23_nlo_as_0119   --pdf_dir $dir_path/MG5_aMC_v2_8_3_2/HEPTools/lhapdf6_py3/share/LHAPDF --db eval_madgraph_1_$f --experiment_name madgraph_$f --config $dir_path/benchmark_01/madgraph_default1.yaml --debug --lhapdf_dir $dir_path/MG5_aMC_v2_8_3_2/HEPTools/lhapdf6_py3/lib/python3.*/site-packages --pdf --cuda 3 --n_search 20 --not-stratified
        mv eval_madgraph_1_$f $dir_path/

        cd ..;
    fi
done
cd ..
cd ..

cd ./fortran_output_ucg/SubProcesses
for f in *;
    do   if [ -d "$f" ];
        then cd $f;
        make matrix2py.so 2>&1> f2py_log.txt 2>&1;
        rm -f matrix2py.so
        mv matrix2py* matrix2py.so
        mkdir -p $benchmark_path/utils/integrands/mg/$f/param
        cp $dir_path/fortran_output_ucg/Cards/param_card.dat $benchmark_path/utils/integrands/mg/$f/param
        cp {nexternal.inc,pmass.inc,param.log} $benchmark_path/utils/integrands/mg/$f/param
        cp matrix2py.so $benchmark_path/utils/integrands/mg/$f
        python3 $benchmark_path/benchmark_madgraph.py --process $f --pdf_type NNPDF23_nlo_as_0119   --pdf_dir $dir_path/MG5_aMC_v2_8_3_2/HEPTools/lhapdf6_py3/share/LHAPDF --db eval_madgraph_5_$f --experiment_name madgraph_$f --config $dir_path/benchmark_01/madgraph_default5.yaml --debug --lhapdf_dir $dir_path/MG5_aMC_v2_8_3_2/HEPTools/lhapdf6_py3/lib/python3.*/site-packages --pdf --cuda 3 --n_search 20 --not-stratified
        mv eval_madgraph_5_$f $dir_path/

        cd ..;
    fi
done
cd ..
cd ..


cd ./fortran_output_ucg/SubProcesses
for f in *;
    do   if [ -d "$f" ];
        then cd $f;
        make matrix2py.so 2>&1> f2py_log.txt 2>&1;
        rm -f matrix2py.so
        mv matrix2py* matrix2py.so
        mkdir -p $benchmark_path/utils/integrands/mg/$f/param
        cp $dir_path/fortran_output_ucg/Cards/param_card.dat $benchmark_path/utils/integrands/mg/$f/param
        cp {nexternal.inc,pmass.inc,param.log} $benchmark_path/utils/integrands/mg/$f/param
        cp matrix2py.so $benchmark_path/utils/integrands/mg/$f
        python3 $benchmark_path/benchmark_madgraph.py --process $f --pdf_type NNPDF23_nlo_as_0119   --pdf_dir $dir_path/MG5_aMC_v2_8_3_2/HEPTools/lhapdf6_py3/share/LHAPDF --db eval_madgraph_10_$f --experiment_name madgraph_$f --config $dir_path/benchmark_01/madgraph_default10.yaml --debug --lhapdf_dir $dir_path/MG5_aMC_v2_8_3_2/HEPTools/lhapdf6_py3/lib/python3.*/site-packages --pdf --cuda 3 --n_search 20 --not-stratified
        mv eval_madgraph_10_$f $dir_path/

        cd ..;
    fi
done
cd ..
cd ..


cd ./fortran_output_ucg/SubProcesses
for f in *;
    do   if [ -d "$f" ];
        then cd $f;
        make matrix2py.so 2>&1> f2py_log.txt 2>&1;
        rm -f matrix2py.so
        mv matrix2py* matrix2py.so
        mkdir -p $benchmark_path/utils/integrands/mg/$f/param
        cp $dir_path/fortran_output_ucg/Cards/param_card.dat $benchmark_path/utils/integrands/mg/$f/param
        cp {nexternal.inc,pmass.inc,param.log} $benchmark_path/utils/integrands/mg/$f/param
        cp matrix2py.so $benchmark_path/utils/integrands/mg/$f
        python3 $benchmark_path/benchmark_madgraph.py --process $f --pdf_type NNPDF23_nlo_as_0119   --pdf_dir $dir_path/MG5_aMC_v2_8_3_2/HEPTools/lhapdf6_py3/share/LHAPDF --db eval_madgraph_20_$f --experiment_name madgraph_$f --config $dir_path/benchmark_01/madgraph_default20.yaml --debug --lhapdf_dir $dir_path/MG5_aMC_v2_8_3_2/HEPTools/lhapdf6_py3/lib/python3.*/site-packages --pdf --cuda 3 --n_search 20 --not-stratified
        mv eval_madgraph_20_$f $dir_path/

        cd ..;
    fi
done
cd ..
cd ..
