# python main.py \
#     --task long --seg --anticipate --pos_emb\
#     --n_query 20 --n_encoder_layer 4 --n_decoder_layer 2 --batch_size 8 --hidden_dim 512 \
#     --dataset 50salads --max_pos_len 3100 --sample_rate 6 --epochs 70 --mode=train --input_type=i3d_transcript --mamba True --split=$1 

python main.py \
    --task long --seg --anticipate --pos_emb\
    --n_query 20 --n_encoder_layer 4 --n_decoder_layer 2 --batch_size 8 --hidden_dim 512 \
    --dataset 50salads --max_pos_len 3100 --sample_rate 6 --epochs 100 --mode=train --input_type=i3d_transcript --mamba True --split=$1 