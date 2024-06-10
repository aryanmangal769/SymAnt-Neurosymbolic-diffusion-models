echo "split1"
python main.py --hidden_dim 128 --n_encoder_layer 2 --n_decoder_layer 1 \
    --seg --task long --anticipate --pos_emb \
    --predict --model=transformer --mode=train --input_type=i3d_transcript --context_dim 128 --split=$1
