out_dir=../models/ssd_mobilenet_v2_raccoon/
mkdir -p $out_dir
python model_main_tf2.py --alsologtostderr --model_dir=$out_dir --checkpoint_every_n=500  \
                         --pipeline_config_path=../models/ssd_mobilenet_v2_raccoon.config \
                         --eval_on_train_data 2>&1 | tee $out_dir/train.log
