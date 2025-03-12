def rename_keys(state_dict, encoder_only=False):
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if encoder_only and not key.startswith("head"):
            key = "segformer.encoder." + key
        if key.startswith("backbone"):
            key = key.replace("backbone", "segformer.encoder")
        if "patch_embed" in key:
            # replace for example patch_embed1 by patch_embeddings.0
            idx = key[key.find("patch_embed") + len("patch_embed")]
            key = key.replace(f"patch_embed{idx}", f"patch_embeddings.{int(idx)-1}")
        if "norm" in key:
            key = key.replace("norm", "layer_norm")
        if "segformer.encoder.layer_norm" in key:
            # replace for example layer_norm1 by layer_norm.0
            idx = key[key.find("segformer.encoder.layer_norm") + len("segformer.encoder.layer_norm")]
            key = key.replace(f"layer_norm{idx}", f"layer_norm.{int(idx)-1}")
        if "layer_norm1" in key:
            key = key.replace("layer_norm1", "layer_norm_1")
        if "layer_norm2" in key:
            key = key.replace("layer_norm2", "layer_norm_2")
        if "block" in key:
            # replace for example block1 by block.0
            idx = key[key.find("block") + len("block")]
            key = key.replace(f"block{idx}", f"block.{int(idx)-1}")
        if "attn.q" in key:
            key = key.replace("attn.q", "attention.self.query")
        if "attn.proj" in key:
            key = key.replace("attn.proj", "attention.output.dense")
        if "attn" in key:
            key = key.replace("attn", "attention.self")
        if "fc1" in key:
            key = key.replace("fc1", "dense1")
        if "fc2" in key:
            key = key.replace("fc2", "dense2")
        if "linear_pred" in key:
            key = key.replace("linear_pred", "classifier")
        if "linear_fuse" in key:
            key = key.replace("linear_fuse.conv", "linear_fuse")
            key = key.replace("linear_fuse.bn", "batch_norm")
        if "linear_c" in key:
            # replace for example linear_c4 by linear_c.3
            idx = key[key.find("linear_c") + len("linear_c")]
            key = key.replace(f"linear_c{idx}", f"linear_c.{int(idx)-1}")
        if key.startswith("head"):
            key = key.replace("head", "classifier")
        new_state_dict[key] = value

    return new_state_dict