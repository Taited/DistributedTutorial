python3 cifar10.py
python3 cifar10_dp.py
python3 -m torch.distributed.launch --nproc_per_node=2 cifar10_ddp.py
python3 -m torch.distributed.launch --nproc_per_node=2 cifar10_ddp_syncBN.py 