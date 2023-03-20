export CUDA_VISIBLE_DEVICES=2
for i in 1 2 3 4
do
  python run_main.py \
  --num_episode 2000 \
  --patience 1000 \
  --dev_interval 200 \
  --model_name bert-spc \
  --output_par_dir one_outputs \
  --dataset fewshot_mams_3way_$i \
  --polarities "negative" "neutral" "positive" \
  --shots 1 \
  --query_size 10
done