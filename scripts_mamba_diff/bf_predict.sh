echo "split1"
python main.py --hidden_dim 128 --n_encoder_layer 4 --n_decoder_layer 1 \
    --seg --task long --anticipate --pos_emb \
    --predict --model=transformer --mode=train --input_type=i3d_transcript --context_dim 128 --context_out_net_h_size 128 --mamba True --split=$1 
