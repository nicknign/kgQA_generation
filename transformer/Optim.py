'''A wrapper class for optimizer '''
import numpy as np

def warmup_linear(x,warmup):
    if x<warmup:
        return x/warmup
    return max((x-1.)/(warmup-1.),0)

class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, d_model, n_warmup_steps, n_train_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        #self.init_lr = np.power(d_model, -0.5)
        self.init_lr = 0.0001
        self.n_train_steps = n_train_steps

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_current_steps += 1
        #lr = self.init_lr * self._get_lr_scale()
        lr = self.init_lr*warmup_linear(self.n_current_steps/self.n_train_steps,self.n_warmup_steps)
        # if self.n_warmup_steps:
        #     warmup_steps_float = float(self.n_warmup_steps)
        #     current_step_float = float(self.n_current_steps)
        #     warmup_percent_done = current_step_float/warmup_steps_float
        #     warmup_learning_rate = self.init_lr*warmup_percent_done
        #     is_warmup = 1.0 if self.n_current_steps<self.n_warmup_steps else 0.0
        #     lr = ((1.0-is_warmup)*lr+is_warmup*warmup_learning_rate)
        for param_group in self._optimizer.param_groups:
            #lr = self.init_lr * warmup_linear(self.n_current_steps / self.n_train_steps, self.n_warmup_steps)
            param_group['lr'] = lr

            # if param_group['name']=='decoder':
            #     #print('decoder lr:',param_group['lr'])
            #     lr = 0.01*warmup_linear(self.n_current_steps / self.n_train_steps, self.n_warmup_steps)
            #     param_group['lr'] = lr
            # elif param_group['name']=='encoder':
            #     #print('encoder lr:', param_group['lr'])
            #     lr = 0.0001 * warmup_linear(self.n_current_steps / self.n_train_steps, self.n_warmup_steps)
            #     param_group['lr'] = lr
