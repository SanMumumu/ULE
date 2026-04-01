CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main_rgb.py \
    --n_gpus 8 --batch_size 8 --num_workers 16 --model DiT-B\
    --data_folder /mnt/data/wangsen/SyncVP/data --output ./results_analysis

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main_rgb.py \
    --n_gpus 8 --batch_size 8 --num_workers 16 \
    --model DiT-S --vae_config configs/model.yaml \
    --data_folder /mnt/nodestor/ws/UniWM/DATA/data --output ./results-S



echo "QWer1234." | sshfs -p 39115 wangsen@218.200.126.238:/mnt/data/wangsen/TriFlow/results_e2e /mnt/nodestor/ws/UniWM/ULE/results_remote/115 -o password_stdin,allow_other
echo "iairws" | sshfs -p 39117 wangsen@218.200.126.238:/mnt/data/wangsen/World_model/ULE/results_e2e /mnt/nodestor/ws/UniWM/ULE/results_remote/117 -o password_stdin,allow_other
tensorboard --logdir_spec=TriFlow_115:/mnt/nodestor/ws/UniWM/ULE/results_remote/115,UniWM_Local:/mnt/nodestor/ws/UniWM/ULE/results_woCFG,WorldModel_117:/mnt/nodestor/ws/UniWM/ULE/results_remote/117 --port 6006 

tensorboard --logdir=/mnt/nodestor/ws/UniWM/ULE/results --port 7008

CUDA_VISIBLE_DEVICES=4 python eval_benchmark.py \
    --config configs/model.yaml \
    --ckpt results/20260317_131005_CITYSCAPES_RGB_42/ckpt_0150000.pt \
    --future_frames 28 \
    --NFE 50 \
    --traj 1 \
    --model DiT-L \
    --data_folder /mnt/nodestor/ws/UniWM/DATA/data \
    --output ./results_benchmark