from src.models.mountings.networks_class import *
from src.models.mountings.objectives_class import *

def get_network(network_name):
    maps = {
        'TCNAE': TcnAE,
        'LATAE': LATTransformerAE,

        'GRUEn': GRUEncoder,
        'LSTMEn': LSTMEncoder,
        'TCNEn': TCNEncoder,
        'ConvSeqEn': ConvSeqEncoder,
        'TransformerEn': TransformerEncoder,
        'CDTTransformerEn': CDTTransformerEncoder,
        'LATTransformerEn': LATTransformerEncoder,
    }

    if network_name in maps.keys():
        return maps[network_name]
    else:
        raise NotImplementedError(f'network is not supported. '
                                  f'please use network structure in {maps.keys()}')


def get_objectives(objective_name):
    maps = {
        'OC': OCLoss,
        'MSE': MSELoss
    }
    if objective_name in maps.keys():
        return maps[objective_name]
    else:
        raise NotImplementedError(f'loss is not supported. '
                                  f'please use objectives in {maps.keys()}')

