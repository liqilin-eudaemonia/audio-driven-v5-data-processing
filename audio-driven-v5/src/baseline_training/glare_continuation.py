"""Continue the completed warmup/cosine schedule with a lower-rate cosine tail."""
import math


class ContinuationScheduler:
    def __init__(self, original, initial_steps, total_steps, start_lr, end_lr):
        if total_steps <= initial_steps or not 0 < end_lr <= start_lr:
            raise ValueError('Expected an extended budget and positive decreasing rates')
        self.original = original
        self.initial_steps = initial_steps
        self.total_steps = total_steps
        self.start_lr = start_lr
        self.end_lr = end_lr

    @property
    def current_step(self):
        return self.original.current_step

    @current_step.setter
    def current_step(self, value):
        self.original.current_step = value

    def step(self):
        if self.current_step < self.initial_steps:
            self.original.step()
            return
        self.current_step += 1
        progress = min(1., (self.current_step - self.initial_steps) /
                       (self.total_steps - self.initial_steps))
        rate = self.end_lr + (self.start_lr - self.end_lr) * .5 * (1 + math.cos(math.pi * progress))
        for group in self.original.optimizer.param_groups:
            group['lr'] = rate
