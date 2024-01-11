class Learner(object):
    def log_update(self, o, a, r, op, logpb, dist, done):
        self.log(o, a, r, op, logpb, dist, done)
        info0 = {'learned': False}
        if self.learn_time(done):
            info = self.learn()
            self.post_learn()
            info0.update(info)
            info0['learned'] = True
        return info0

    def log(self, o, a, r, op, logpb, dist, done):
        pass

    def learn_time(self, done):
        pass

    def post_learn(self):
        pass

    def learn(self, env=None):
        pass