KERAS_BACKEND=theano python sample_autoencoder.py \
    ../data/best_vae_model.json \
    ../data/best_vae_annealed_weights.ht \
    ../data/250k_rndm_zinc_drugs_clean.smi \
    ../data/zinc_char_list.json \
    -l5000
