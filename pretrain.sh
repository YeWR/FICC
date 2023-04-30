#envs=(breakout pong ms_pacman)
env=$1

python -u pretrain.py \
      --dataset ${env} \
      --model_name ficc_${env} \
      --device 0 \
      --batch_size 128 \
      --extra_info none \
      --weight_decay 0.01 \
      --l1_penalty 0.01 \
      --dataset_path ./atari_replay_dataset/
