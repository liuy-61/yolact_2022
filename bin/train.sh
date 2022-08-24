export CUDA_VISIBLE_DEVICES=0,1
python train.py --dataset=coco_person_dataset --config=yolact_base_config --batch_size=8 --num_epochs=100  --exp_name=warm_up_100_epohs_with_imageNet_weights
