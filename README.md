# street_view_name_extraction

A modified version Google's of attention based OCR for streetview feature extraction (https://github.com/tensorflow/models/tree/master/research/attention_ocr). More details about the mode's architecture can be found here ((https://arxiv.org/abs/1704.03549)

# Environment

The model was run on Ubuntu 16.04, python3 and trained using a GPU.

# Pre processing

Generate the tensorflow record following the FSNS format (https://stackoverflow.com/questions/44430310/how-to-create-dataset-in-the-same-format-as-the-fsns-dataset/44461910#44461910)
Place the images and labels in /data folder create a dic.txt that has the mapping between the characters and number.

Move the created records to /data/fsns/train/ and modify python/datasets/newtextdataset.py to fit your dataset.

# Training

To train the model, run:
```
python train.py --dataset_name=newtextdataset
```
# Using the model
```
python demo_inference.py --batch_size=32 \
  --checkpoint=model.ckpt-399731\
  --image_path_pattern=./datasets/data/fsns/temp/fsns_train_%02d.png  
```
