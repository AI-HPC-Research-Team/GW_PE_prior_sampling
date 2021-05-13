
python -m lfigw.gwpe_extra train new nde \
    --data_dir waveforms/GW150914_posterior/ \
    --model_dir models/GW150914_posterior/ \
    --basis_dir waveforms/GW150914/ \
    --nbins 8 \
    --dont_sample_extrinsic_only \
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
