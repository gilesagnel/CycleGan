"""
This class implements the CycleGAN model, which facilitates image-to-image translation for 
unpaired data. For more information, refer to the 
CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
"""

import torch
import itertools
from models.generator import Generator
from models.generator_res import ResnetGenerator
from models.discriminator import Discriminator
from utils.image_pool import ImagePool
from models.loss import GANLoss
from torch.nn import init
from torch.optim import lr_scheduler
import os
from collections import OrderedDict


class CycleGan():
    def __init__(self, opt):
        self.opt = opt
        self.isTrain = opt.isTrain
        self.pool_size = opt.image_pool_size
        self.device = opt.device
        self.save_dir = opt.save_dir
        self.metric = 0
        
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        self.visual_names = ['real_A', 'fake_B', 'rec_A', 'real_B', 'fake_A', 'rec_B']
        
        if self.isTrain and self.opt.lambda_identity > 0.0: 
            self.visual_names.insert(3, 'idt_A') 
            self.visual_names.insert(7, 'idt_B') 

        self.model_names = ['G_A', 'G_B']
        if self.isTrain:
            self.model_names.extend(['D_A', 'D_B'])


        self.netG_A = ResnetGenerator(self.opt.input_nc, self.opt.output_nc, self.opt.ngf, 9)
        init_net(self.netG_A, self.opt.init_type, device=self.device)
        self.netG_B = ResnetGenerator(self.opt.output_nc, self.opt.input_nc, self.opt.ngf, 9)
        init_net(self.netG_B, self.opt.init_type, device=self.device)
        
        if self.isTrain:
            self.netD_A = Discriminator(self.opt.output_nc, self.opt.ndf)
            init_net(self.netD_A, self.opt.init_type, device=self.device)
            self.netD_B = Discriminator(self.opt.input_nc, self.opt.ndf)
            init_net(self.netD_B, self.opt.init_type, device=self.device)

            if self.opt.lambda_identity > 0.0:  
                assert(self.opt.input_nc == self.opt.output_nc)
            self.fake_A_pool = ImagePool(self.pool_size)  
            self.fake_B_pool = ImagePool(self.pool_size)  

            self.criterionGAN = GANLoss(self.opt.gan_mode).to(self.device)  
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.optimizer_G = torch.optim.Adam(
                itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), 
                lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(
                itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), 
                lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            self.optimizers = [self.optimizer_G, self.optimizer_D]

    def setup(self):
        if self.isTrain:
            self.schedulers = [get_scheduler(optimizer, self.opt.lr_policy, self.opt.n_epochs, 
                                             self.opt.n_epochs_decay) for optimizer in self.optimizers]
        
        if not self.isTrain or self.opt.continue_train:
            load_suffix = f'iter_{self.opt.load_iter}' if self.opt.load_iter > 0 else \
                  f'{self.opt.load_epoch}' if self.opt.load_epoch > 0 else \
                  "latest"
            self.load_networks(load_suffix)
        self.print_networks(self.opt.verbose)

    def print_networks(self, verbose):
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def load_networks(self, epoch):
        for name in self.model_names:
            if self.device == "cuda":
                device = "cuda"
            else:
                device = torch.device(self.device)
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                state_dict = torch.load(load_path)
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata
                net.load_state_dict(state_dict)
                net.to(device)

    def update_learning_rate(self):
        old_lr = self.optimizers[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate %.7f -> %.7f' % (old_lr, lr))

    def get_current_visuals(self):
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret
    
    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret
    
    def save_networks(self, epoch):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = f"{epoch}_net_{name}.pth"
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, f"net{name}")

                if self.device in ["cuda", "mps"]:
                    net = net.cpu()

                torch.save(net.state_dict(), save_path)

                if self.device in ["cuda", "mps"]:
                    net.to(self.device)

    def set_input(self, input):
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.image_paths = input['A_paths']

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def eval(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()
    
    def get_image_paths(self):
        return self.image_paths

    def forward(self):
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))

    def test(self):
        with torch.no_grad():
            self.forward()

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        with torch.no_grad():
            pred_fake = netD(fake)
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        lambda_idt = self.opt.lambda_identity
        lambda_A =self.opt.lambda_A
        lambda_B = self.opt.lambda_B

        if lambda_idt > 0:
            # Identity loss
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = self.loss_idt_B = 0


        # GAN loss
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)

        # Cycle consistency loss
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

        # Total loss and backpropagation
        self.loss_G = (self.loss_G_A + self.loss_G_B + 
                    self.loss_cycle_A + self.loss_cycle_B + 
                    self.loss_idt_A + self.loss_idt_B)

        self.loss_G.backward()

    def optimize_parameters(self):
        # Forward pass
        self.forward()

        # Update generators (G_A and G_B)
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        # Update discriminators (D_A and D_B)
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        self.optimizer_D.step()


def init_net(net, init_type='normal', init_gain=0.02, device='cpu'):
    if device == 'mps' or device =='cuda':
        mps_device = torch.device(device)
        net.to(mps_device)
        device_ids = [0]
        net = torch.nn.DataParallel(net, device_ids)  

    def init_func(m): 
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

    return net


def get_scheduler(optimizer, lr_policy, n_epochs, n_epochs_decay=None,
                  lr_decay_iters=None):
    if lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 - n_epochs) / float(n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay_iters, gamma=0.1)
    elif lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', lr_policy)
    return scheduler
