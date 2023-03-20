export CUDA_VISIBLE_DEVICES=0
python run_3way_com_bert.py \
--num_episode 1000 \
--patience 500 \
--dev_interval 500 \
--model_name bert-spc \
--output_par_dir hard_test_outputs \
--dataset fewshot_14rest_hard_3way \
--polarities "negative" "positive" "neutral" \
--shots 1,5 \
--query_size 1