# @TODO: ZXH (add the network structure of LAT)


class LATTransformerEncoder:
    def __init__(self, n_features, n_output=20, seq_len=100, d_model=128,
                 n_heads=8, n_hidden='128', dropout=0.1,
                 attn='cc_attn', token_encoding='convolutional', pos_encoding='fixed',
                 activation='GELU', bias=False,
                 norm='LayerNorm', freeze=False):
        return


class LATTransformerAE:
    def __init__(self, n_features, n_output=20, seq_len=100, d_model=128,
                 n_heads=8, n_hidden='128', dropout=0.1,
                 attn='cc_attn', token_encoding='convolutional', pos_encoding='fixed',
                 activation='GELU', bias=False,
                 norm='LayerNorm', freeze=False):
        return