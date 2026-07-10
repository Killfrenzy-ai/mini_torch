class LRScheduler:

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.last_epoch = -1

    def get_lr(self):
        raise NotImplementedError

    def step(self):
        self.last_epoch += 1

        self.optimizer.lr = self.get_lr()

        return self.optimizer.lr