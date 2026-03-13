CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main_rgb.py --n_gpus 8 --batch_size 8 --num_workers 12 --output ./results_e2e

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main_rgb.py --n_gpus 8 --batch_size 8 --output ./results_CFG


echo "QWer1234." | sshfs -p 39115 wangsen@218.200.126.238:/mnt/data/wangsen/TriFlow/results_e2e /mnt/nodestor/ws/UniWM/ULE/results_remote/115 -o password_stdin,allow_other
echo "iairws" | sshfs -p 39117 wangsen@218.200.126.238:/mnt/data/wangsen/World_model/ULE/results_e2e /mnt/nodestor/ws/UniWM/ULE/results_remote/117 -o password_stdin,allow_other
tensorboard --logdir_spec=TriFlow_115:/mnt/nodestor/ws/UniWM/ULE/results_remote/115,UniWM_Local:/mnt/nodestor/ws/UniWM/ULE/results_woCFG,WorldModel_117:/mnt/nodestor/ws/UniWM/ULE/results_remote/117 --port 6006 