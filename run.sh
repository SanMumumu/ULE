# /mnt/nodestor/ws/UniWM/DATA/data  /mnt/data/wangsen/SyncVP/data 

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main_rgb.py \
    --n_gpus 8 --batch_size 8 --num_workers 16 \
    --data_folder /mnt/data/wangsen/SyncVP/data --output ./results_disc

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main_rgb.py \
    --n_gpus 8 --batch_size 32 --num_workers 16 \
    --data_folder /mnt/nodestor/ws/UniWM/DATA/data --output ./results_disc

# 清理挂载点
fusermount -u /mnt/nodestor/ws/UniWM/ULE/results_remote


echo "QWer1234." | sshfs -p 39115 wangsen@218.200.126.238:/mnt/data/wangsen/ULWM/ULE/results_wo_disc /mnt/nodestor/ws/UniWM/ULE/results_remote -o password_stdin,allow_other
echo "iairws" | sshfs -p 39117 wangsen@218.200.126.238:/mnt/data/wangsen/World_model/ULE/results_disc /mnt/nodestor/ws/UniWM/ULE/results_remote -o password_stdin,allow_other
tensorboard --logdir_spec=wo_disc:/mnt/nodestor/ws/UniWM/ULE/results_remote/,disc:/mnt/nodestor/ws/UniWM/ULE/results_remote/,BigBS:/mnt/nodestor/ws/UniWM/ULE/results_woCFG --port 6006



python eval_benchmark.py \
    --config configs/model.yaml \
    --ckpt results_woCFG/20260313_033450_CITYSCAPES_RGB_42/ckpt_0110000.pt \
    --future_frames 28 \
    --NFE 50 \
    --traj 1 \
    --output ./results_benchmark_110k