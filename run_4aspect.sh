export CUDA_VISIBLE_DEVICES=0
python run_2way_4aspect.py \
--num_episode 5000 \
--model_name aspect \
--output_par_dir 4aspect_test_outputs \
--dataset fewshot_mams_2way \
--polarities "negative" "positive" \
--shots 1,5,10 \
--query_size 10