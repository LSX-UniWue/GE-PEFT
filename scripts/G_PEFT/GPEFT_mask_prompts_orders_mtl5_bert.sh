export CUDA_VISIBLE_DEVICES=0

bs=8
dropout=0.1
sl=256
psl=20
epoch=40
n_train_total=115000
n_val_per_class=500
lamb=10

A=agnews
B=yelp
C=amazon
D=yahoo
E=dbpedia
lr_A=1e-5
lr_B=1e-5
lr_C=1e-5
lr_D=1e-5
lr_E=1e-5
for seed in 0 1 2
    do
        task_order=${A}_${B}_${C}_${D}_${E}
        lr_list=${lr_A}_${lr_B}_${lr_C}_${lr_D}_${lr_E}

        echo $task_order
        echo $lr_list
        python3 src/run_continual_masked_mtl5_bert.py \
            --model_name_or_path bert-base-uncased \
            --mtl_task_list $task_order \
            --continual_learning \
            --mask_prompts \
            --query_encoder_type avg_all_embed \
            --task_specific_classifier \
            --direct_compose False \
            --softmax_match_scale 8 \
            --matching_loss_v2 \
            --key_init_func uniform \
            --prompt_init_func uniform \
            --do_train \
            --do_eval \
            --do_predict \
            --early_stop \
            --early_stopping_patience 5 \
            --max_seq_length $sl \
            --per_device_train_batch_size $bs \
            --learning_rate_list $lr_list \
            --num_train_epochs $epoch \
            --pre_seq_len $psl \
            --n_train_total ${n_train_total} \
            --n_val_per_class ${n_val_per_class} \
            --output_dir checkpoints_continual_mask_mtl5_bert/order1_mask_prompts \
            --overwrite_output_dir \
            --hidden_dropout_prob $dropout \
            --seed $seed \
            --save_strategy epoch \
            --evaluation_strategy epoch \
            --prefix \
            --lamb $lamb

        task_order=${B}_${C}_${D}_${E}_${A}
        lr_list=${lr_B}_${lr_C}_${lr_D}_${lr_E}_${lr_A}
        python3 src/run_continual_masked_mtl5_bert.py \
            --model_name_or_path bert-base-uncased \
            --mtl_task_list $task_order \
            --continual_learning \
            --mask_prompts \
            --query_encoder_type avg_all_embed \
            --task_specific_classifier \
            --direct_compose False \
            --softmax_match_scale 8 \
            --matching_loss_v2 \
            --key_init_func uniform \
            --prompt_init_func uniform \
            --do_train \
            --do_eval \
            --do_predict \
            --early_stop \
            --early_stopping_patience 5 \
            --max_seq_length $sl \
            --per_device_train_batch_size $bs \
            --learning_rate_list $lr_list \
            --num_train_epochs $epoch \
            --pre_seq_len $psl \
            --n_train_total ${n_train_total} \
            --n_val_per_class ${n_val_per_class} \
            --output_dir checkpoints_continual_mask_mtl5_bert/order2_mask_prompts \
            --overwrite_output_dir \
            --hidden_dropout_prob $dropout \
            --seed $seed \
            --save_strategy epoch \
            --evaluation_strategy epoch \
            --prefix \
            --lamb $lamb

        
        task_order=${E}_${D}_${A}_${C}_${B}
        lr_list=${lr_E}_${lr_D}_${lr_A}_${lr_C}_${lr_B}
        python3 src/run_continual_masked_mtl5_bert.py \
            --model_name_or_path bert-base-uncased \
            --mtl_task_list $task_order \
            --continual_learning \
            --mask_prompts \
            --query_encoder_type avg_all_embed \
            --task_specific_classifier \
            --direct_compose False \
            --softmax_match_scale 8 \
            --matching_loss_v2 \
            --key_init_func uniform \
            --prompt_init_func uniform \
            --do_train \
            --do_eval \
            --do_predict \
            --early_stop \
            --early_stopping_patience 5 \
            --max_seq_length $sl \
            --per_device_train_batch_size $bs \
            --learning_rate_list $lr_list \
            --num_train_epochs $epoch \
            --pre_seq_len $psl \
            --n_train_total ${n_train_total} \
            --n_val_per_class ${n_val_per_class} \
            --output_dir checkpoints_continual_mask_mtl5_bert/order3_mask_prompts \
            --overwrite_output_dir \
            --hidden_dropout_prob $dropout \
            --seed $seed \
            --save_strategy epoch \
            --evaluation_strategy epoch \
            --prefix \
            --lamb $lamb

        
        task_order=${B}_${A}_${E}_${C}_${D}
        lr_list=${lr_B}_${lr_A}_${lr_E}_${lr_C}_${lr_D}
        python3 src/run_continual_masked_mtl5_bert.py \
            --model_name_or_path bert-base-uncased \
            --mtl_task_list $task_order \
            --continual_learning \
            --mask_prompts \
            --query_encoder_type avg_all_embed \
            --task_specific_classifier \
            --direct_compose False \
            --softmax_match_scale 8 \
            --matching_loss_v2 \
            --key_init_func uniform \
            --prompt_init_func uniform \
            --do_train \
            --do_eval \
            --do_predict \
            --early_stop \
            --early_stopping_patience 5 \
            --max_seq_length $sl \
            --per_device_train_batch_size $bs \
            --learning_rate_list $lr_list \
            --num_train_epochs $epoch \
            --pre_seq_len $psl \
            --n_train_total ${n_train_total} \
            --n_val_per_class ${n_val_per_class} \
            --output_dir checkpoints_continual_mask_mtl5_bert/order4_mask_prompts \
            --overwrite_output_dir \
            --hidden_dropout_prob $dropout \
            --seed $seed \
            --save_strategy epoch \
            --evaluation_strategy epoch \
            --prefix \
            --lamb $lamb
    done
