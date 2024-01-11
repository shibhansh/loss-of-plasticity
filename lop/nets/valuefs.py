from torch import nn
from lop.utils.miscellaneous import net_init, fc_body, register_hook, kaiming_init, xavier_init, lecun_init


class VF(object):
    def value(self, x, to_log_features=False):
        self.to_log_features = to_log_features
        val = self.v_net(x)
        self.to_log_features = False
        return val


class MLPVF(VF, nn.Module):
    def __init__(self, o_dim, act_type='Tanh', h_dim=(50,), device='cpu', init='kaiming'):
        super().__init__()
        self.act_type = act_type
        self.device = device
        self.v_net = fc_body(act_type, o_dim, h_dim)
        if len(h_dim) > 0:
            self.v_net.append(nn.Linear(h_dim[-1], 1))
        else:
            self.v_net.append(nn.Linear(o_dim, 1))
        self.v_net = nn.Sequential(*self.v_net)
        if init == 'kaiming':
            kaiming_init(self.v_net, act=act_type.lower())
        elif init == 'xavier':
            xavier_init(self.v_net, act=act_type)
        elif init == 'lecun':
            lecun_init(self.v_net)
        elif init == 'default':
            net_init(self.v_net)
        self.to(device)
        # Setup feature logging
        self.setup_feature_logging(h_dim=h_dim)

    def setup_feature_logging(self, h_dim) -> None:
        self.to_log_features = False
        # Prepare for logging
        self.activations = {}
        self.feature_keys = [self.v_net[i * 2 + 1] for i in range(len(h_dim))]

        def hook_fn(m, i, o):
            if self.to_log_features:
                self.activations[m] = o

        # Register "hook_fn" for each layer
        register_hook(self.v_net, hook_fn)

    def get_activations(self,):
        return [self.activations[key] for key in self.feature_keys]