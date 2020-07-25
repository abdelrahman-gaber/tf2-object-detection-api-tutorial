DATASET_PATH=../data/raccoon_data
OUTPUT_CSV_TRAIN=../data/raccoon_data/train_labels.csv
OUTPUT_CSV_VAL=../data/raccoon_data/test_labels.csv

# 1. generate csv files from xml annotations
# train
python xml_to_csv.py --annot_dir $DATASET_PATH/train/annotations --out_csv_path $OUTPUT_CSV_TRAIN
# validation
python xml_to_csv.py --annot_dir $DATASET_PATH/test/annotations --out_csv_path $OUTPUT_CSV_VAL

# 2. generate tfrecords for training object detection
# train
python generate_tfrecord.py --path_to_images $DATASET_PATH/train/images --path_to_annot $OUTPUT_CSV_TRAIN \
                            --path_to_label_map ../models/raccoon_labelmap.pbtxt \
                            --path_to_save_tfrecords $DATASET_PATH/train.record
# validation
python generate_tfrecord.py --path_to_images $DATASET_PATH/test/images --path_to_annot $OUTPUT_CSV_VAL \
                            --path_to_label_map ../models/raccoon_labelmap.pbtxt \
                            --path_to_save_tfrecords $DATASET_PATH/val.record
