"""
This class abstracts the creation of target label tensors matching input sizes, encapsulating
 various GAN objectives
"""
import torch as T
from torch import nn

class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', T.tensor(target_real_label))
        self.register_buffer('fake_label', T.tensor(target_fake_label))
        loss_functions = {
        'lsgan': nn.MSELoss(),
        'vanilla': nn.BCEWithLogitsLoss(),
        'wgangp': None
        }

        self.loss = loss_functions.get(gan_mode, None)

    def get_target_tensor(self, prediction, target_is_real):
        return (self.real_label if target_is_real else self.fake_label) \
                .expand_as(prediction)
    
    def __call__(self, prediction, target_is_real):
        if self.loss is None:
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        else:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        return loss