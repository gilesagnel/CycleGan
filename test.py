import os
from utils.options import TestOptions
from dataset.dataset import UnPairedDataset
from models.model import CycleGan
from utils import util
from utils.visualizer import save_images
import torch

if __name__ == '__main__':
    opt = TestOptions()    
    phase = 'train' if opt.isTrain else 'val'
    dataloader = torch.utils.data.DataLoader(
            UnPairedDataset(opt),
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.num_threads))
    model = CycleGan(opt)       
    model.setup()               

    web_dir = os.path.join(opt.checkpoints_dir, opt.model_name, '{}_{}'.format(phase, opt.epoch_count))  
    if opt.load_iter > 0:  
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = util.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.model_name, phase, opt.epoch_count))
    model.eval()
    for i, data in enumerate(dataloader):
        if i > 100:
            break
        model.set_input(data)   
        model.test()            
        visuals = model.get_current_visuals()  
        img_path = model.get_image_paths()     
        if i % 10 == 0:  
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=1, width=opt.win_size, use_wandb=opt.use_wandb)
    webpage.save()  
