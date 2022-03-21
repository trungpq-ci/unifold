#! /bin/bash
python3 generate_pkl_features.py \
    --fasta_dir ./test_fasta \
    --output_dir ./out \
    --data_dir /home/ubuntu/openfold/openfold-main/data \
    --num_workers 1

