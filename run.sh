# vae (ideally one per modality)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main_rgb.py --n_gpus 8 --batch_size 8 --num_workers 12 --output ./results_e2e
# single modality  model
python3 main.py --config configs/run/train/ddpm_city_rgb.yaml


tensorboard --logdir=/mnt/data/wangsen/TriFlow/results_e2e/20260306_002640_CITYSCAPES_RGB_42 --port=7004