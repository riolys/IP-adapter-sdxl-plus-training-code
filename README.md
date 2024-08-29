# IP-adapter-sdxl-plus-training-code
Since there is no official code for IP-adapter sdxl plus, we produce the process here.

In line344, the image_pro_model should be modified for sdxl as follows:
'''
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
'''
