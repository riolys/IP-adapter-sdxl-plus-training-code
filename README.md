# IP-adapter-sdxl-plus-training-code
Since there is no official code for IP-adapter sdxl plus, we produce the process here.

In line 344, the image_pro_model should be modified for sdxl as follows:
```
image_proj_model = Resampler(
    dim=1280,
    depth=4,
    dim_head=64,
    heads=20,
    num_queries=16,
    embedding_dim=image_encoder.config.hidden_size,
    output_dim=2048,
    ff_mult=4,
    )
```
In line 439, the image_embeds should be the -2 layer feature of clip(vit-H) text encoder 
```
image_embeds = image_encoder(batch["clip_images"].to(accelerator.device, dtype=weight_dtype), output_hidden_states=True).hidden_states[-2]
```
Before training, ensure your json file for your dataset is as the same format as in line 58.
```
text = item['text']
image_file = item['image']
```

Do not forget that your image encoder path is clip(vit-H) not clip(vit-G)!!
```
accelerate launch --num_processes 8  --multi_gpu  --mixed_precision "fp16" \
  tutorial_train_sdxl.py \
  --pretrained_model_name_or_path="stable-diffusion-xl-base-1.0" \
  --image_encoder_path="IP-Adapter/models/image_encoder" \
  --pretrained_ip_adapter_path="IP-Adapter/sdxl_models/ip-adapter-plus_sdxl_vit-h.bin" \
  --data_json_file="IP-Adapter/train.json" \
  --data_root_path="" \
  --mixed_precision="fp16" \
  --resolution=512 \
  --train_batch_size=8 \
  --dataloader_num_workers=4 \
  --learning_rate=1e-04 \
  --weight_decay=0.01 \
  --output_dir="sd-ip_adapter_p" \
  --save_steps=10000
```
