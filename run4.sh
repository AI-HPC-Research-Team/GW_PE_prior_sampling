python3.7 -m inference.gwpe_main train new nde \
    --data_dir data/GW150914_sample_prior_posterior_basis/ \
    --model_dir models/GW150914_sample_posterior_100basis_extrinsic_posterior_prior/ \
    --basis_dir data/GW150914_sample_prior_posterior_basis/ \
    --save_model_name model.pt \
    --save_aux_filename waveforms_supplementary.hdf5 \
    --nbins 8 \
    --nsamples_target_event 500 \
    --nsample 100000 \
    --sampling_from posterior \
    --num_transform_blocks 10 \
    --nflows 15 \
    --batch_norm \
    --batch_size 2048 \
    --lr 0.0002 \
    --epochs 1000 \
    --distance_prior_fn uniform_distance \
    --hidden_dims 512 \
    --truncate_basis 100 \
    --activation elu \
    --lr_anneal_method cosine
# train data sampling from posterior
# for extrinsic params dim
# using posterior basis (truncate 100)