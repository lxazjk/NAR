import os
import torch

from nanovllm import NARSamplingParams, NARPagedLLM


def main():
    # Use the repo's pretrained checkpoint by default.
    ckpt = os.environ.get(
        "NAR_GPT_CKPT",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pretrained_models", "c2i_B_256.pt"),
    )

    llm = NARPagedLLM(
        ckpt,
        block_size=256,
        cls_token_num=1,
        num_classes=1000,
        device="cuda",
        dtype=torch.bfloat16,
        # GPT-B preset
        dim=768,
        n_layer=12,
        n_head=12,
        medusa_attention_num=1,
    )
    
    sampling_params = NARSamplingParams(
        temperature=1.0,
        top_k=200,
        top_p=0.9,
        cfg_scale=4.0,
        cfg_interval=-1,
        block_size=256,
        model_type='c2i',
    )
    
    class_labels = torch.tensor([207, 360, 388, 889])
    
    generated_tokens = llm.generate_image(
        condition=class_labels,
        sampling_params=sampling_params,
        use_tqdm=True,
    )
    
    print(f"Generated tokens shape: {generated_tokens.shape}")
    print(f"Generated tokens: {generated_tokens}")


if __name__ == "__main__":
    main()
