export PYTHONPATH=../../../:$PYTHONPATH
CUDA_VISIBLE_DEVICES=$4
python -m torch.distributed.run --nproc_per_node=$1 ../../../tools/train_val.py -e
