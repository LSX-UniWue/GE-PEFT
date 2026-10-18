export CUDA_VISIBLE_DEVICES=0

bs=8
lr=1e-5
dropout=0.1
sl=256
psl=16
exp=4
epoch=40
early_stopping_patience=5

for seed in 0 1 2
    do
        python3 src/run_continual_growonly_mtl_wos.py \
            --model_name_or_path bert-base-uncased \
            --mtl_task_list 0_1_2_3_4_5_6 \
            --continual_learning \
            --mask_prompts False\
            --expand_prompts True\
            --task_specific_classifier \
            --matching_loss_v2 \
            --do_train \
            --do_eval \
            --do_predict \
            --early_stop \
            --early_stopping_patience ${early_stopping_patience} \
            --max_seq_length $sl \
            --per_device_train_batch_size $bs \
            --learning_rate $lr \
            --num_train_epochs $epoch \
            --pre_seq_len $psl \
            --output_dir checkpoints_continual_expand_mtl_wos/continual_expand_prompts \
            --overwrite_output_dir \
            --hidden_dropout_prob $dropout \
            --seed $seed \
            --save_strategy epoch \
            --evaluation_strategy epoch \
            --expansion_size $exp \
            --prefix
    done
