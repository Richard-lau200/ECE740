# echo "../../../:$PYTHONPATH"
export PYTHONPATH=../../../:$PYTHONPATH
# export PYTHONPATH= /home/jupyter-tliu13@ualberta.ca-0beb7/UniAD/experiments/CIFAR-10/01234/:$PYTHONPATH

CUDA_VISIBLE_DEVICES=$4
python -m torch.distributed.run --nproc_per_node=$1 ../../../tools/train_val.py
