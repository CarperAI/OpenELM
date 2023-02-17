import torch
from transformers import CodeGenForCausalLM, GPTJConfig, GPTJForCausalLM


def cg2gptj(model_name):
    print(f"Model we're converting: {model_name}")
    cg_model = CodeGenForCausalLM.from_pretrained(
        model_name, use_cache=True, torch_dtype="auto"
    )
    cg_config = cg_model.config

    # Create empty GPTJ model
    print("Creating empty GPTJ model")
    config = GPTJConfig(
        vocab_size=cg_config.vocab_size,
        n_positions=cg_config.n_positions,
        n_embd=cg_config.n_embd,
        n_layer=cg_config.n_layer,
        n_head=cg_config.n_head,
        rotary_dim=cg_config.rotary_dim,
        n_inner=cg_config.n_inner,
        activation_function=cg_config.activation_function,
        resid_pdrop=cg_config.resid_pdrop,
        embd_pdrop=cg_config.embd_pdrop,
        attn_pdrop=cg_config.attn_pdrop,
        layer_norm_epsilon=cg_config.layer_norm_epsilon,
        initializer_range=cg_config.initializer_range,
        scale_attn_weights=cg_config.scale_attn_weights,
        use_cache=cg_config.use_cache,
        bos_token_id=cg_config.bos_token_id,
        eos_token_id=cg_config.eos_token_id,
        torch_dtype=cg_config.torch_dtype,
    )
    # Fix tokenizer type
    config.tokenizer_class = "CodeGenTokenizer"

    gptj_model = GPTJForCausalLM(config)
    embed_dim = config.n_embd

    def replace(model, weights, name):
        model.state_dict()[name].copy_(weights.detach())

    def replace_by_name(dest_model, src_model, old_name, new_name):
        assert old_name in src_model.state_dict()
        assert new_name in dest_model.state_dict()
        replace(dest_model, src_model.state_dict()[old_name], new_name)

    print("Converting...")
    # Copy weights from CodeGen model
    with torch.no_grad():
        cg_model.eval()
        gptj_model.eval()

        for name, param in cg_model.named_parameters():
            # print(f'Converting {name}')
            # Handle the qkv weights separately because we need to split them
            if "qkv_proj" in name:
                qkv_proj = param.detach().clone()
                mp_num = 4  # number of cores on their TPU I guess?
                local_dim = embed_dim // mp_num
                # GPT-J and CodeGen slice up the qkv projection slightly differently.
                # After a great deal of pain, I figured out that this permutation on
                # the weights of the qkv_proj fixes it.
                base_permutation = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]
                permutation = torch.cat(
                    [
                        torch.arange(i * local_dim, (i + 1) * local_dim)
                        for i in base_permutation
                    ]
                )
                # NB: we permute the *rows* here because the computation is xA.T
                new_qkv_proj = qkv_proj[permutation, :]
                # NB: the name QKV is misleading here; they are actually stored in
                #     the order QVK
                query, value, key = torch.split(new_qkv_proj, embed_dim, dim=0)
                replace(gptj_model, query, name.replace("qkv_proj", "q_proj"))
                replace(gptj_model, key, name.replace("qkv_proj", "k_proj"))
                replace(gptj_model, value, name.replace("qkv_proj", "v_proj"))
            else:
                replace_by_name(gptj_model, cg_model, name, name)

    return gptj_model
