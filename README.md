# SSBU_Classify

This project is intended to analyze pro player's movement in Super Smash Bros Ultimate from video. 

## Environment setting
In this project, tenserflow detection API is used and installed from this [tutorial](https://tensorflow-object-detection-api-tutorial.readthedocs.io).

Make sure the CUDA and toolkit are installed on your computer in order to use GPU to train the model. Once installed, check gpu setup with installtest.py
+ Platform: Windows 10 
+ Tensorflow verion: 2.8.0
+ CUDA version: 11.6
+ Python version: 3.9

## Dataset
Modify PATH in video2frame.py into your own recording, generate target data set.
```bash
python split_train_test.py
```
All data is labelled with [labelImg tool](https://github.com/tzutalin/labelImg) 

For simplicity reasons, the test character is ike. Initially a match recording was used to do a simple target detection for testing tenserflow detection API. While the model performed well under a single label, it did not perform well in a classification task that included almost most of moves (about 40 labels). This is because moves like neutral B are not used much and the sample size is insufficient.

Thus, in the dataset each move is recorded back and front 2-4 times. (3k frames in total)

Then, split them into train/test set.
```bash
python partition_dataset.py -x -i D:\smash\dect\tf-detection\dataset\classify_2\ -r 0.1
```

Creat TFRecords
```bash
python generate_tfrecord.py -x D:\smash\dect\tf-detection\dataset\classify_2\train -l D:\smash\dect\tf-detection\dataset\classify_2/label_map.pbtxt -o D:\smash\dect\tf-detection\dataset\classify_2/train.record

python generate_tfrecord.py -x D:\smash\dect\tf-detection\dataset\classify_2\train -l D:\smash\dect\tf-detection\dataset\classify_2/label_map.pbtxt -o D:\smash\dect\tf-detection\dataset\classify_2/test.record
```
## Training Guide

Download pre-trained model from [Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md). In this project, Faster R-CNN ResNet50 is used after comparing performance in a simple training task. Modify pipeline.config with your own training setting.

Here the restriction of GPU memory is added into model_main_tf2.py to avoid allocating error.  Train the model with the command

```bash
python model_main_tf2.py --model_dir=./class_frcnn_2 --pipeline_config_path=./class_frcnn_2/pipeline.config 
```

And if you want to evaluate the model performance in the same time. I recommand you to run evaluation on CPU 

```bash
set CUDA_VISIBLE_DEVICES=-1 
python model_main_tf2.py --model_dir=./class_frcnn_2 --pipeline_config_path=./class_frcnn_2/pipeline.config --checkpoint_dir=./class_frcnn_2
```

## Export Data
Export a trained model 
```bash
python .\exporter_main_v2.py --input_type image_tensor --pipeline_config_path .\class_frcnn_2\pipeline.config --trained_checkpoint_dir .\class_frcnn_2\ --output_directory .\exported-models\class_frcnn_2_20k
```

Then you are able to get the interface with getinterface.py. Also, you could export the data into a csv file by uncommenting last few lines.

## Discussion
[From dataset](doc/fromdataset.gif)

In this project, I believe that the most difficult step is set up your custom dataset. As you can see in the above gif, trained model works well in trained data. However, its performance is not as good as expected in other recordings. 

[From other clip](doc/otherclip.gif)

This can be caused by a variety of reasons:

1. The main problem is that the data size is not enough. 3k is too small for such a project.

2. Ike may not be a ideal research material. As you can see I add label 'landing' in the annotation. It's because so many moves of ike start or end with crounching. In order not to confuse the subject action, all these lags are count into landing.

3. More noticeable movement are needed for this project. For Ike, there is little difference between Nair and Bair's starting because he only have one sword. So maybe Belyth is a better target in this kind of project.


