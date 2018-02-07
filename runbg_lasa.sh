mkdir experiments/$1
echo $1 > ../current_experiment
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,nvcc.flags=-D_FORCE_INLINES,lib.cnmem=1 python main_data_generation.py
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,nvcc.flags=-D_FORCE_INLINES,lib.cnmem=1 stdbuf -oL nohup python experiment_brain_parcellation_lasa.py $1 > experiments/$1/log.log 2>&1&
echo $! > save_pid.txt