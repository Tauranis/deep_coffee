echo "Data augmentation for good beans"
docker run \
    -v ${PWD}/dataset:/dataset \
    --rm --gpus all deep_coffee \
    python -m deep_coffee.image_proc.data_aug \
    --input_dir /dataset/good \
    --output_dir /dataset/good \
    --angle_list 45,90,135,180,225,270

echo "Data augmentation for bad beans"
docker run \
    -v ${PWD}/dataset:/dataset \
    --rm --gpus all deep_coffee \
    python -m deep_coffee.image_proc.data_aug \
    --input_dir /dataset/bad \
    --output_dir /dataset/bad \
    --angle_list 45,90,135,180,225,270

echo "Create tfrecords"
docker run \
    -v ${PWD}/dataset:/dataset \
    --rm --gpus all deep_coffee \
    python -m deep_coffee.ml.images_to_tfrecords \
    --output_dir /dataset/tfrecords \
    --tft_artifacts_dir /dataset/tft_artifacts \
    --good_beans_dir /dataset/good \
    --good_beans_list_train /dataset/protocol/good_train.txt \
    --good_beans_list_eval /dataset/protocol/good_eval.txt \
    --good_beans_list_test /dataset/protocol/good_test.txt \
    --bad_beans_dir /dataset/bad \
    --bad_beans_list_train /dataset/protocol/bad_train.txt \
    --bad_beans_list_eval /dataset/protocol/bad_eval.txt \
    --bad_beans_list_test /dataset/protocol/bad_test.txt \
    --image_dim 224 \
    --n_shards 20 \
    --ext jpg \
    --network mobilenet \
    --temp-dir /tmp
