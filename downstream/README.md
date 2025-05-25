# SurgSSL

***
## Model

### Hyper-parameter
```c
qkv_bias: The bias of QKV Conv
qkv_divided_bias: QKV Conv without bias, and additional bias
qkv_divided: The employment of Q/K/V Conv
patch_embed_2d: Use 2D Conv for patch embedding
```

### Code
`downstream/model/unifiedmodel.py`
> ***For Video-level Parameter***
```c

# The only difference between these two models is the additional parameter “st”, 
# which corresponds to different Position Embedding initialization strategies.
# So unified_base_st is for parameters pretrained by SurgMAEKD/MVD, 
# and unified_base is for parameters pretrained by SurgMAE/VideoMAE/VideoMAEV2/UMT/MAE.

unified_base_st: get_3d_sincos_pos_embed() from mvd/modeling_student.py
unified_base: get_sinusoid_encoding_table() from VideoMAE/modeling_pretrain.py
```

> ***For Image-level Parameter***
> According our extensive experiments, “unified_base_2D” achieves optimal performance using half the length of clip.
```c
# The only difference between these three models is how to initialize the patch embedding, 
# which corresponds to different Patch Embedding architectures (2D Conv/ 3D Conv).

unified_base_2D: directly use 2D Conv for patch embedding, initialed from image-level parameters
unified_base_2D_3D: use 3D Conv for patch embedding, extending 2D Conv parameters to 3D Conv
unified_base: directly use 3D Conv for patch embedding and initialize from scratch
unified_base_st: for different position embedding
```

`downstream/model/unifiedmodel_cls.py`
> This code is designed to read image-level parameters with learnable position embedding and cls token. However, according to extensive experiments, the performance is not as good as directly abandoning cls token and learnable position embedding, so this model is abandoned.

`downstream/model/timesformer_vmae.py`
> The code logic is consistent with unifiedmodel.py, replacing the jointly spatial-temporal attention with divided spatial-temporal attention.

`downstream/model/timesformer.py`
> code of TimeSformer, modified from orignial code.

### Parameters
#### Video-level Parameters
`pretrain_params/videomae_base_k400_1600e.pth`
> Kinetics-400 (unified_base)
> Tong, Z., Song, Y., Wang, J., & Wang, L. (2022). Videomae: Masked autoencoders are data-efficient learners for self-supervised video pre-training. Advances in neural information processing systems, 35, 10078-10093.
> https://github.com/MCG-NJU/VideoMAE/blob/main/MODEL_ZOO.md
> https://drive.google.com/file/d/1tEhLyskjb755TJ65ptsrafUG2llSwQE1/view?usp=sharing

`pretrain_params/videomae_base_ssv2_2400e.pth`
> Something-Something V2
> Tong, Z., Song, Y., Wang, J., & Wang, L. (2022). Videomae: Masked autoencoders are data-efficient learners for self-supervised video pre-training. Advances in neural information processing systems, 35, 10078-10093.
> https://github.com/MCG-NJU/VideoMAE/blob/main/MODEL_ZOO.md
> https://drive.google.com/file/d/1tEhLyskjb755TJ65ptsrafUG2llSwQE1/view?usp=sharing


`pretrain_params/internvideo_base_hybrid_800e.pth`
> Hybrid (unified_base)
> Wang, Y., Li, K., Li, Y., He, Y., Huang, B., Zhao, Z., ... & Qiao, Y. (2022). Internvideo: General video foundation models via generative and discriminative learning. arXiv preprint arXiv:2212.03191.
> https://github.com/OpenGVLab/InternVideo/tree/main/InternVideo1/Pretrain/VideoMAE

`pretrain_params/videomaev2_base_ft_k710_pt_hybrid_giant.pth`
> Hybrid-pretraining + K710-finetuning (unified_base)
> Wang, L., Huang, B., Zhao, Z., Tong, Z., He, Y., Wang, Y., ... & Qiao, Y. (2023). Videomae v2: Scaling video masked autoencoders with dual masking. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 14549-14560).
> https://github.com/OpenGVLab/VideoMAEv2/blob/master/docs/MODEL_ZOO.md#Distillation

`pretrain_params/umt_base_k710_200e.pth`
> Kinetics-710 (unified_base)
> Li, K., Wang, Y., Li, Y., Wang, Y., He, Y., Wang, L., & Qiao, Y. (2023). Unmasked teacher: Towards training-efficient video foundation models. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 19948-19960).
> https://huggingface.co/OpenGVLab/UMT/tree/main

`pretrain_params/mvd_base_k400_400e.pth`
> Kinetics-400 (unified_base_st)
> Wang, R., Chen, D., Wu, Z., Chen, Y., Dai, X., Liu, M., ... & Jiang, Y. G. (2023). Masked video distillation: Rethinking masked feature modeling for self-supervised video representation learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 6312-6322).
> https://github.com/ruiwang2021/mvd/blob/main/MODEL_ZOO.md
> https://drive.google.com/file/d/1MFgpUuHTtFMiormgenmxypU9gqRDbbwb/view?usp=sharing

#### Image-level Parameters
`pretrain_params/mae_base_imagenet1k.bin`
> ImageNet1K-MAE (unified_base_2D)
> He, K., Chen, X., Xie, S., Li, Y., Dollár, P., & Girshick, R. (2022). Masked autoencoders are scalable vision learners. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 16000-16009).
> https://huggingface.co/timm/vit_base_patch16_224.mae

`pretrain_params/dino_base_imagenet1k.bin`
> ImageNet1K-DINO (unified_base_2D)
> Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J., Bojanowski, P., & Joulin, A. (2021). Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision (pp. 9650-9660).
> https://huggingface.co/timm/vit_base_patch16_224.dino

`pretrain_params/surpervised_base_imagenet1k.bin`
> ImageNet1K (unified_base_2D)
> Dosovitskiy, A. (2020). An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929.
> https://huggingface.co/timm/vit_base_patch16_224.augreg_in1k/

`pretrain_params/surpervised_base_imagenet21k.bin`
> ImageNet21K (unified_base_2D)
> Dosovitskiy, A. (2020). An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929.
> https://huggingface.co/timm/vit_base_patch16_224.orig_in21k

`pretrain_params/clip_base_wit400m.bin`
> WIT400M (unified_base_2D)
> Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., ... & Sutskever, I. (2021, July). Learning transferable visual models from natural language supervision. In International conference on machine learning (pp. 8748-8763). PMLR.
> https://huggingface.co/timm/vit_base_patch16_clip_224.openai