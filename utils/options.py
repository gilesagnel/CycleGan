from dataclasses import dataclass, field
import os


@dataclass
class Options:
    data_root: str = field(init=False)
    isTrain: bool = False
    model_name: str = "selfie2anime"
    device: str = "mps"

    lambda_identity: float = 0.5
    lambda_A: float = 10.0
    lambda_B: float = 10.0

    input_nc: int = 3
    output_nc: int = 3
    ngf: int = 64
    ndf: int = 64
    image_pool_size: int = 50
    
    gan_mode: str = "lsgan"
    init_type: str = "normal"

    lr: float = 2e-4
    beta1: float = 0.5
    lr_policy: str = 'linear'
    epoch_count: int = 0
    n_epochs: int = 50
    n_epochs_decay: int = 50
    continue_train: bool = True
    load_iter: int = 0
    load_epoch: int = 0
    print_freq: int = 50
    batch_size: int = 1
    display_freq: int = 50
    update_html_freq: int = 1000
    save_latest_freq: int = 5000
    save_by_iter: bool = False
    save_epoch_freq: int = 5
    serial_batches: bool = False
    num_threads: int = 8

    # Directories
    checkpoints_dir: str = './checkpoints'
    save_dir: str = field(init=False)

    # Output options
    verbose: bool = False
    crop_size: int = 256
    load_size: int = 286
    preprocess: bool = True

    # Visualization options
    display_id: int = 1
    no_html: bool = False
    win_size: int = 256
    display_port: int = 8097
    use_wandb: bool = False
    wandb_project_name: str = None
    current_epoch: int = 0
    display_ncols: int = 4
    display_server: str = 'http://localhost'
    display_env: str = 'main'

    def __post_init__(self):
        self.data_root = f"./dataset/{self.model_name}"
        self.save_dir = os.path.join(self.checkpoints_dir, self.model_name)


@dataclass
class TrainOptions(Options):
    isTrain: bool = True
    continue_train: bool = False
    epoch_count: int = 0
    load_epoch: int = 0

@dataclass
class TestOptions(Options):
    isTrain: bool = False
    load_epoch: int = 0
    num_threads = 0   
    batch_size = 1    
    serial_batches = True  
    display_id = -1   