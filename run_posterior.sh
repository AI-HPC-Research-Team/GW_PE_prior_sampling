
python3.7 -m lfigw.gwpe_extra train new nde \
    --data_dir waveforms/GW170729_posterior/ \
    --model_dir models/GW170729_posterior/ \
    --nbins 8 \
    --num_transform_blocks 10 \
    --nflows 15 \
    --batch_norm \
    --batch_size 2048 \
    --lr 0.0002 \
    --epochs 500 \
    --distance_prior_fn uniform_distance \
    --hidden_dims 512 \
    --truncate_basis 100 \
    --activation elu \
    --lr_anneal_method cosine
