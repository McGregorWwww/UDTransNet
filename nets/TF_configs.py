import ml_collections

def get_model_config():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.transformer.num_heads  = 4
    config.transformer.num_layers = 4
    config.expand_ratio           = 2
    config.transformer.embedding_channels = 32 * config.transformer.num_heads
    config.KV_size = config.transformer.embedding_channels * 4
    config.KV_size_S = config.transformer.embedding_channels
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0.1
    config.patch_sizes=[16,8,4,2]
    config.base_channel = 32
    config.decoder_channels = [32,64,128,256,512]
    return config

