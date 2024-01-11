import torch


class Agent(object):
    def __init__(self, pol, learner, device='cpu', to_log_features=False):
        self.pol = pol
        self.learner = learner
        self.device = device
        self.to_log_features = to_log_features

    def get_action(self, o):
        """
        :param o: np. array of shape (1,)
        :return: a two tuple
        - np.array of shape (1,)
        - np.array of shape (1,)
        """
        action, lprob, dist = self.pol.action(torch.tensor(o, dtype=torch.float32, device=self.device).unsqueeze(0),
                                              to_log_features=self.to_log_features)
        features = None
        if self.to_log_features:
            features = self.pol.get_activations()
        return action[0].cpu().numpy(), lprob.cpu().numpy(), self.pol.dist_to(dist, to_device='cpu'), features

    def log_update(self, o, a, r, op, logp, dist, done):
        return self.learner.log_update(o, a, r, op, logp, dist, done)

    def preprocess_state(self, state):
        return state

    def choose_action(self, o, epsilon):
        return self.get_action(o=o)[0]