echo $$ > rel_models_data.pid

python run_finetuning_nli_1gpu.py \
          --pre_model_path "roberta-large" \
          --train_folder "../../../data/train/" \
          --dev_folder "../../../data/dev/" \
          --test_folder "../../../data/test/" \
          --train_file "train-temporal-relation-data.tsv" \
          --dev_file "dev-temporal-relation-data.tsv" \
          --test_file "test-temporal-relation-data.tsv" \
          --num_labels 2 \
          --hypothesis_only 'False' \
          --train_batch_size 16 \
          --num_train_epochs 2 \
          --learning_rate '2e-5' \
          --weight_decay 0.1 \
          --warmup_steps 122 \
          --logging_steps 5000 \
          --save_steps 5000

python run_finetuning_nli_1gpu.py \
          --pre_model_path "roberta-large" \
          --train_folder "../../../data/train/" \
          --dev_folder "../../../data/dev/" \
          --test_folder "../../../data/test/" \
          --train_file "train-temporal-relation-data.tsv" \
          --dev_file "dev-temporal-relation-data.tsv" \
          --test_file "test-temporal-relation-data.tsv" \
          --num_labels 2 \
          --hypothesis_only 'True' \
          --train_batch_size 16 \
          --num_train_epochs 2 \
          --learning_rate '2e-5' \
          --weight_decay 0.1 \
          --warmup_steps 122 \
          --logging_steps 5000 \
          --save_steps 5000
