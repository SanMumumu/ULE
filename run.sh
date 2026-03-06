CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main_rgb.py --n_gpus 8 --batch_size 8 --num_workers 12 --output ./results_e2e

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main_rgb.py --n_gpus 8 --batch_size 8 --output ./results_e2e


tensorboard --logdir=/mnt/data/wangsen/TriFlow/results_e2e/20260306_032003_CITYSCAPES_RGB_42 --port=7014