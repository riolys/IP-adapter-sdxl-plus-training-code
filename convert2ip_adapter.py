import os
import torch

def convert(src_dir):


    ckpt = os.path.join(src_dir, "pytorch_model.bin")
    #sd = torch.load(ckpt)
    sd = torch.load(ckpt, weights_only=True)

    image_proj_sd = {}
    ip_sd = {}
    countIpK = 1
    countIpV = 1
    #countIpQ = 0

    countToQLoraUp = 0
    countToQLoraDown = 0
    countToKLoraUp = 0
    countToKLoraDown = 0
    countToVLoraUp = 0
    countToVLoraDown = 0

    countToOutLoraUp = 0
    countToOutLoraDown = 0

    ## corresct one
    ip_ckpt = "IP-Adapter/sdxl_models/ip-adapter-plus_sdxl_vit-h.bin"
    ip_dict = torch.load(ip_ckpt, map_location="cpu")["ip_adapter"]
    ##

    for k in sd:
        if k.startswith("image_proj_model"):
            image_proj_sd[k.replace("image_proj_model.", "")] = sd[k]

    for k in sd:
        if "down_" in k:
            if "to_k_ip." in k:
                ip_sd[str(countIpK) + '.to_k_ip.weight'] = sd[k]
                countIpK += 2
            elif "to_v_ip." in k:
                ip_sd[str(countIpV) + '.to_v_ip.weight'] = sd[k]
                countIpV += 2

    for k in sd:
        if "up_" in k:
            if "to_k_ip." in k:
                ip_sd[str(countIpK) + '.to_k_ip.weight'] = sd[k]
                countIpK += 2
            elif "to_v_ip." in k:
                ip_sd[str(countIpV) + '.to_v_ip.weight'] = sd[k]
                countIpV += 2

    for k in sd:
        if "mid_" in k:
            if "to_k_ip." in k:
                ip_sd[str(countIpK) + '.to_k_ip.weight'] = sd[k]
                countIpK += 2
            elif "to_v_ip." in k:
                ip_sd[str(countIpV) + '.to_v_ip.weight'] = sd[k]
                countIpV += 2

    for k in sd:
        if "to_q_lora.down" in k:
            ip_sd[str(countToQLoraDown) + ".to_q_lora.down.weight"] = sd[k]
            countToQLoraDown += 1
        elif "to_k_lora.down" in k:
            ip_sd[str(countToKLoraDown) + ".to_k_lora.down.weight"] = sd[k]
            countToKLoraDown += 1
        elif "to_v_lora.down" in k:
            ip_sd[str(countToVLoraDown) + ".to_v_lora.down.weight"] = sd[k]
            countToVLoraDown += 1
        elif "to_out_lora.down" in k:
            ip_sd[str(countToOutLoraDown) + ".to_out_lora.down.weight"] = sd[k]
            countToOutLoraDown += 1
        elif "to_q_lora.up" in k:
            ip_sd[str(countToQLoraUp) + ".to_q_lora.up.weight"] = sd[k]
            countToQLoraUp += 1
        elif "to_k_lora.up" in k:
            ip_sd[str(countToKLoraUp) + ".to_k_lora.up.weight"] = sd[k]
            countToKLoraUp += 1
        elif "to_v_lora.up" in k:
            ip_sd[str(countToVLoraUp) + ".to_v_lora.up.weight"] = sd[k]
            countToVLoraUp += 1
        elif "to_out_lora.up" in k:
            ip_sd[str(countToOutLoraUp) + ".to_out_lora.up.weight"] = sd[k]
            countToOutLoraUp += 1

    res = {}
    list = {'to_k_ip', 'to_q_ip', 'to_v_ip', "to_q_lora", "to_k_lora", "to_v_lora","to_out_lora" }
    for e in list:
        res[e] = 0
    for k,v in ip_dict.items():
        for e in list:
            if e in k:
                res[e] += 1
                break

    seta = set()
    setb = set()
    for k,v in ip_dict.items():
        seta.add(k)
    for k,v in ip_sd.items():
        setb.add(k)
    useless = sorted(setb - seta)

    for e in useless:
        ip_sd.pop(e)
    torch.save({"image_proj": image_proj_sd, "ip_adapter": ip_sd}, os.path.join(src_dir, "ip_adapter.bin"))

convert("IP-Adapter/sd-ip_adapter_ppp/checkpoint-20")