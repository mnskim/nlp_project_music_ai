export CUDA_VISIBLE_DEVICES=3
lrs=(1e-3 1e-4 1e-5 1e-6)
n_layers=(3 4 5)
bsz=(16 32)
dims=(128 256 512)
drs=(0.1 0.3 0.5)

for bs in "${bsz[@]}"
do
for lr in "${lrs[@]}"
    do
        for dim in "${dims[@]}"
        do
            for n_layer in "${n_layers[@]}"
            do
                for dr in "${drs[@]}"
                do
                    python lstm_classifier.py --do_train \
                    --hidden_dim ${dim} \
                    --lr ${lr} \
                    --num_layers ${n_layer} \
                    --dropout ${dr} \
                    --batch_size ${bs} \
                    --output_dir output/lstm_bs${bsz}_lr${lr}_dim${dim}_nlayer${n_layer}_dr${dr}_bs${bs}
                done
            done
        done
    done
done