import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))


def load_nar_model_from_gpt(model: nn.Module, ckpt_path: str):
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "module" in checkpoint:
        state_dict = checkpoint["module"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    
    gpt_to_nar_mapping = {
        "tok_embeddings.weight": "model.embed_tokens.weight",
        "cls_embedding.embedding_table.weight": "model.cls_embedding.embedding_table.weight",
        "norm.weight": "model.norm.weight",
        "medusa_norm.weight": "model.medusa_norm.weight",
        "output.weight": "lm_head.weight",
        "medusa_output.weight": "medusa_head.weight",
    }
    
    for i in range(100):
        gpt_to_nar_mapping[f"layers.{i}.attention.wqkv.weight"] = f"model.layers.{i}.self_attn.qkv_proj.weight"
        gpt_to_nar_mapping[f"layers.{i}.attention.wo.weight"] = f"model.layers.{i}.self_attn.o_proj.weight"
        gpt_to_nar_mapping[f"layers.{i}.attention_norm.weight"] = f"model.layers.{i}.input_layernorm.weight"
        gpt_to_nar_mapping[f"layers.{i}.feed_forward.w1.weight"] = f"model.layers.{i}.mlp.gate_up_proj.weight"
        gpt_to_nar_mapping[f"layers.{i}.feed_forward.w2.weight"] = f"model.layers.{i}.mlp.down_proj.weight"
        gpt_to_nar_mapping[f"layers.{i}.feed_forward.w3.weight"] = None
        gpt_to_nar_mapping[f"layers.{i}.ffn_norm.weight"] = f"model.layers.{i}.post_attention_layernorm.weight"
    
    new_state_dict = {}
    for gpt_name, param in state_dict.items():
        nar_name = None
        
        if gpt_name in gpt_to_nar_mapping:
            nar_name = gpt_to_nar_mapping[gpt_name]
        else:
            for gpt_key, nar_key in gpt_to_nar_mapping.items():
                if gpt_key.replace('.weight', '') in gpt_name:
                    if nar_key is not None:
                        nar_name = gpt_name.replace(gpt_key.replace('.weight', ''), nar_key.replace('.weight', ''))
                    break
        
        if nar_name is not None:
            if ".w1.weight" in gpt_name:
                base_name = nar_name.replace(".w1.weight", ".gate_up_proj.weight")
                if base_name not in new_state_dict:
                    w1_weight = param
                    w3_name = gpt_name.replace(".w1.weight", ".w3.weight")
                    w3_weight = state_dict.get(w3_name, torch.zeros_like(w1_weight))
                    gate_up_weight = torch.cat([w1_weight, w3_weight], dim=0)
                    new_state_dict[base_name] = gate_up_weight
            elif ".w3.weight" in gpt_name:
                pass
            elif ".wqkv.weight" in gpt_name:
                wqkv_weight = param
                out_features, in_features = wqkv_weight.shape
                n_head = model.config.n_head
                n_kv_head = model.config.n_kv_head or n_head
                head_dim = in_features // n_head
                
                q_dim = n_head * head_dim
                kv_dim = n_kv_head * head_dim
                
                wq, wk, wv = wqkv_weight.split([q_dim, kv_dim, kv_dim], dim=0)
                qkv_weight = torch.cat([wq, wk, wv], dim=0)
                new_state_dict[nar_name] = qkv_weight
            else:
                new_state_dict[nar_name] = param
    
    model_state_dict = model.state_dict()
    loaded_keys = []
    missing_keys = []
    for name, param in new_state_dict.items():
        if name in model_state_dict:
            if model_state_dict[name].shape == param.shape:
                model_state_dict[name] = param
                loaded_keys.append(name)
            else:
                print(f"Shape mismatch for {name}: model={model_state_dict[name].shape}, loaded={param.shape}")
        else:
            missing_keys.append(name)
    
    model.load_state_dict(model_state_dict, strict=False)
    
    print(f"Loaded {len(loaded_keys)} weights from GPT checkpoint")
    if missing_keys:
        print(f"Missing keys in model: {missing_keys[:5]}...")
    
    return model
