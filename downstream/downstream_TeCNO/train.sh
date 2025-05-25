export CUDA_VISIBLE_DEVICES=1

# python train.py -c modules/cnn/config/config_feature_extract.yml
# python train.py -c modules/SelfSupSurg/config/config_feature_extract.yml
python train.py -c modules/GSViT/config/config_feature_extract.yml
# python train.py -c modules/EndoViT/config/config_feature_extract.yml
# python train.py -c modules/EndoSSL/config/config_feature_extract.yml