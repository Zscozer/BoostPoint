# BoostPoint



## Data Preparation

Firstly, you need to have the shapeNet rendered image data, and we obtain the rendered image according to https://github.com/Xharlie/ShapenetRender_more_variation. Secondly, you need to enter data_utils and run make_render_datasets.py and make_superpixels_datasets.py to create the superpixel features of the rendered image.

## Pre-training

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 train_debias.py --log_dir your_log_dir --batch_size 576 --use_colors --epoch 200 --model choose_your_model
```

## Fine-tuning

After pre-training the model, directly enter the finetune, enter the respective protocol, and load the checkpoints of the pre-trained model.

## Note
Due to webpage upload restrictions, our complete dataset, and all checkpoints will be released after being accepted.
