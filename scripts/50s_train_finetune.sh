python main.py \
    --task long --seg --anticipate --pos_emb\
    --n_query 20 --n_encoder_layer 2 --n_decoder_layer 2 --batch_size 8 --hidden_dim 512 \
    --dataset 50salads --max_pos_len 3100 --sample_rate 6 --epochs 50 --mode=train --input_type=i3d_transcript --context_dim 512 --split=$1 --finetune --lr 0.0001 

