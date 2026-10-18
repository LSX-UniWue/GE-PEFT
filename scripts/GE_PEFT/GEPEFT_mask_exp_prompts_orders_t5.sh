export CUDA_VISIBLE_DEVICES=0

bs=8
dropout=0.1
sl=512
psl=50
exp=12
gpsl=10
epoch=40
lamb=10


A=agnews
C=amazon
D=yahoo
E=dbpedia
lr_A=2e-2
lr_C=5e-2
lr_D=2e-2
lr_E=5e-2



for seed in 0 1 2
    do
        task_list=${E}_${C}_${D}_${A}
        lr_list=${lr_E}_${lr_C}_${lr_D}_${lr_A}
        echo $task_list
        echo $lr_list
        python3 src/run_continual_taskgrowing_mask_mtl5_t5.py \
            --model_name_or_path google-t5/t5-large \
            --mtl_task_list $task_list\
            --continual_learning \
            --mask_prompts \
            --matching_loss_v2 \
            --do_train \
            --do_eval \
            --do_predict \
            --n_train_per_class 16 \
            --n_val_per_class 500 \
            --early_stop \
            --early_stopping_patience 5 \
            --max_seq_length $sl \
            --per_device_train_batch_size $bs \
            --learning_rate_list $lr_list \
            --num_train_epochs $epoch\
            --pre_seq_len $psl \
            --output_dir=checkpoints_continual_mask_exp_mtl5_t5/order1_mask_exp_prompts \
            --overwrite_output_dir \
            --hidden_dropout_prob $dropout \
            --seed $seed \
            --save_strategy epoch \
            --evaluation_strategy epoch  \
            --prefix \
            --expansion_size $exp \
            --lamb $lamb


        
        task_list=${E}_${C}_${A}_${D}
        lr_list=${lr_E}_${lr_C}_${lr_A}_${lr_D}
        python3 src/run_continual_mask_mtl5_t5.py \
            --model_name_or_path google-t5/t5-large \
            --mtl_task_list $task_list\
            --continual_learning \
            --mask_prompts \
            --matching_loss_v2 \
            --do_train \
            --do_eval \
            --do_predict \
            --n_train_per_class 16 \
            --n_val_per_class 500 \
            --early_stop \
            --early_stopping_patience 5 \
            --max_seq_length $sl \
            --per_device_train_batch_size $bs \
            --learning_rate_list $lr_list \
            --num_train_epochs $epoch\
            --pre_seq_len $psl \
            --output_dir=checkpoints_continual_mask_exp_mtl5_t5/order2_mask_exp_prompts \
            --overwrite_output_dir \
            --hidden_dropout_prob $dropout \
            --seed $seed \
            --save_strategy epoch \
            --evaluation_strategy epoch  \
            --prefix \
            --expansion_size $exp \
            --lamb $lamb


        task_list=${D}_${C}_${A}_${E}
        lr_list=${lr_D}_${lr_C}_${lr_A}_${lr_E}
        python3 src/run_continual_mask_mtl5_t5.py \
            --model_name_or_path google-t5/t5-large \
            --mtl_task_list $task_list\
            --continual_learning \
            --mask_prompts \
            --matching_loss_v2 \
            --do_train \
            --do_eval \
            --do_predict \
            --n_train_per_class 16 \
            --n_val_per_class 500 \
            --early_stop \
            --early_stopping_patience 5 \
            --max_seq_length $sl \
            --per_device_train_batch_size $bs \
            --learning_rate_list $lr_list \
            --num_train_epochs $epoch\
            --pre_seq_len $psl \
            --output_dir=checkpoints_continual_mask_exp_mtl5_t5/order3_mask_exp_prompts \
            --overwrite_output_dir \
            --hidden_dropout_prob $dropout \
            --seed $seed \
            --save_strategy epoch \
            --evaluation_strategy epoch  \
            --prefix \
            --expansion_size $exp \
            --lamb $lamb
    done
