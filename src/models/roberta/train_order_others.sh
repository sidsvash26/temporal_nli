echo $$ > rel_models_timebank_data.pid 
#############  red ################
#python run_finetuning_nli.py \
#          --pre_model_path "roberta-large" \
#          --train_folder "../../data/train/" \
#          --dev_folder "../../data/dev/" \
#          --test_folder "../../data/test/" \
#          --train_file "train-red-data.tsv" \
#          --dev_file "dev-red-data.tsv" \
#          --test_file "test-red-data.tsv" \
#          --num_labels 2 \
#          --hypothesis_only 'False' \
#          --train_batch_size 16 \
#          --num_train_epochs 10 \
#          --learning_rate '0.1' \
#          --weight_decay 0.99 \
#          --warmup_steps 122 \
#          --logging_steps 303 \
#          --save_steps 303
#
python run_finetuning_nli.py \
          --pre_model_path "roberta-large" \
          --train_folder "../../data/train/" \
          --dev_folder "../../data/dev/" \
          --test_folder "../../data/test/" \
          --train_file "train-red-data.tsv" \
          --dev_file "dev-red-data.tsv" \
          --test_file "test-red-data.tsv" \
          --num_labels 2 \
          --hypothesis_only 'True' \
          --train_batch_size 16 \
          --num_train_epochs 20 \
          --learning_rate '0.1' \
          --weight_decay 0.99 \
          --warmup_steps 122 \
          --logging_steps 303 \
          --save_steps 303
##############    tbdense    ################
#python run_finetuning_nli.py \
#          --pre_model_path "roberta-large" \
#          --train_folder "../../data/train/" \
#          --dev_folder "../../data/dev/" \
#          --test_folder "../../data/test/" \
#          --train_file "train-tbdense-data.tsv" \
#          --dev_file "dev-tbdense-data.tsv" \
#          --test_file "test-tbdense-data.tsv" \
#          --num_labels 2 \
#          --hypothesis_only 'False' \
#          --train_batch_size 16 \
#          --num_train_epochs 10 \
#          --learning_rate '2e-5' \
#          --weight_decay 0.1 \
#          --warmup_steps 122 \
#          --logging_steps 464 \
#          --save_steps 464
#
#python run_finetuning_nli.py \
#          --pre_model_path "roberta-large" \
#          --train_folder "../../data/train/" \
#          --dev_folder "../../data/dev/" \
#          --test_folder "../../data/test/" \
#          --train_file "train-tbdense-data.tsv" \
#          --dev_file "dev-tbdense-data.tsv" \
#          --test_file "test-tbdense-data.tsv" \
#          --num_labels 2 \
#          --hypothesis_only 'True' \
#          --train_batch_size 16 \
#          --num_train_epochs 10 \
#          --learning_rate '2e-5' \
#          --weight_decay 0.1 \
#          --warmup_steps 122 \
#          --logging_steps 464 \
#          --save_steps 464
#
#################      tempeval3  ################
python run_finetuning_nli.py \
          --pre_model_path "roberta-large" \
          --train_folder "../../data/train/" \
          --dev_folder "../../data/dev/" \
          --test_folder "../../data/test/" \
          --train_file "train-tempeval3-data.tsv" \
          --dev_file "dev-tempeval3-data.tsv" \
          --test_file "test-tempeval3-data.tsv" \
          --num_labels 2 \
          --hypothesis_only 'False' \
          --train_batch_size 16 \
          --num_train_epochs 20 \
          --learning_rate '0.1' \
          --weight_decay 0.99 \
          --warmup_steps 122 \
          --logging_steps 1287 \
          --save_steps 1287

#python run_finetuning_nli.py \
#          --pre_model_path "roberta-large" \
#          --train_folder "../../data/train/" \
#          --dev_folder "../../data/dev/" \
#          --test_folder "../../data/test/" \
#          --train_file "train-tempeval3-data.tsv" \
#          --dev_file "dev-tempeval3-data.tsv" \
#          --test_file "test-tempeval3-data.tsv" \
#          --num_labels 2 \
#          --hypothesis_only 'True' \
#          --train_batch_size 16 \
#          --num_train_epochs 20 \
#          --learning_rate '0.1' \
#          --weight_decay 0.99 \
#          --warmup_steps 122 \
#          --logging_steps 1287 \
#          --save_steps 1287
