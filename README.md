This is a sample tenserflow detection test from the tutorial: https://tensorflow-object-detection-api-tutorial.readthedocs.io
Please set up the environment with above instruction.
Check gpu setup with installtest.py

1. modify PATH in video2frame.py, generate target data set.

2. Label data with lableimg

3. Partition the dataset
python partition_dataset.py -x -i D:\smash\dect\tf-detection\dataset\train\ -r 0.1
python partition_dataset.py -x -i D:\smash\dect\tf-detection\dataset\classify_2\ -r 0.1

4. Create TensorFlow records

python generate_tfrecord.py -x D:\smash\dect\tf-detection\dataset\train\train -l D:\smash\dect\tf-detection\dataset\train/label_map.pbtxt -o D:\smash\dect\tf-detection\dataset\train/train.record

python generate_tfrecord.py -x D:\smash\dect\tf-detection\dataset\train\test -l D:\smash\dect\tf-detection\dataset\train/label_map.pbtxt -o D:\smash\dect\tf-detection\dataset\train/test.record

python generate_tfrecord.py -x D:\smash\dect\tf-detection\dataset\classify_2\train -l D:\smash\dect\tf-detection\dataset\classify_2/label_map.pbtxt -o D:\smash\dect\tf-detection\dataset\classify_2/train.record

python generate_tfrecord.py -x D:\smash\dect\tf-detection\dataset\classify_2\train -l D:\smash\dect\tf-detection\dataset\classify_2/label_map.pbtxt -o D:\smash\dect\tf-detection\dataset\classify_2/test.record

5. Setup the pre-trained model and the config file
Start training process with SSD ResNet50 V1 FPN 640x640 model
python model_main_tf2.py --model_dir=./my_ssd_resnet50_v1_fpn --pipeline_config_path=./my_ssd_resnet50_v1_fpn/pipeline.config 


python model_main_tf2.py --model_dir=./class_frcnn_2 --pipeline_config_path=./class_frcnn_2/pipeline.config --checkpoint_dir=./class_frcnn_2


set CUDA_VISIBLE_DEVICES=-1 
python model_main_tf2.py --model_dir=./my_frcnn --pipeline_config_path=./my_frcnn/pipeline.config --checkpoint_dir=./my_frcnn


python model_main_tf2.py --model_dir=./my_ssd_resnet50_v1_fpn --pipeline_config_path=./my_ssd_resnet50_v1_fpn/pipeline.config --checkpoint_dir=./my_ssd_resnet50_v1_fpn

tensorboard --logdir=my_ssd_resnet50_v1_fpn

tensorboard --logdir=class_frcnn_2

Here the restriction of GPU memory is added into model_main_tf2.py to avoid allocating error. 


6. Export the model

python .\exporter_main_v2.py --input_type image_tensor --pipeline_config_path .\my_ssd_resnet50_v1_fpn\pipeline.config --trained_checkpoint_dir .\my_ssd_resnet50_v1_fpn\ --output_directory .\exported-models\my_model

python .\exporter_main_v2.py --input_type image_tensor --pipeline_config_path .\my_frcnn\pipeline.config --trained_checkpoint_dir .\my_frcnn\ --output_directory .\exported-models\frcnn

python .\exporter_main_v2.py --input_type image_tensor --pipeline_config_path .\class_frcnn\pipeline.config --trained_checkpoint_dir .\class_frcnn\ --output_directory .\exported-models\class_frcnn_20k

7. Get interface with getinterface.py