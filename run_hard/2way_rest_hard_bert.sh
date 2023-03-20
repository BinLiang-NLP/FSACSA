export CUDA_VISIBLE_DEVICES=3
python run_3way_com_bert.py \
--num_episode 1000 \
--patience 500 \
--dev_interval 500 \
--model_name bert-spc \
--output_par_dir hard_test_outputs \
--dataset fewshot_rest_hard_2way \
--polarities "negative" "positive" \
--shots 1,5 \
--query_size 1