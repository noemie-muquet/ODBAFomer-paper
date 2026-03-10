import warnings
from shutil import copyfile
import inspect
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, StepLR
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, DeviceStatsMonitor, Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from omegaconf import OmegaConf
import os
import argparse
from einops import rearrange
from pytorch_lightning import Trainer, seed_everything
from earthformer.config import cfg
from earthformer.utils.optim import SequentialLR, warmup_lambda
from earthformer.utils.utils import get_parameter_names
from earthformer.cuboid_transformer.cuboid_transformer import CuboidTransformerModel
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
from dataloaders.wind_dataloader import ODBADataModule, ODBATestDataModule


# set directories 
_curr_dir = os.path.realpath(os.path.dirname(os.path.realpath(__file__)))
exps_dir = os.path.join(_curr_dir, "experiments")

train_dir = os.path.join(_curr_dir, 'data', 'train_data', 'ODBAFormer_inputs', 'train')
test_dir = os.path.join(_curr_dir, 'data', 'train_data', 'ODBAFormer_inputs', 'eval')

if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')

   
class CuboidODBAModule(pl.LightningModule):
    
    def __init__(self,
                 total_num_steps: int,
                 oc_file: str = None,
                 save_dir: str = None,
                 test =  False):
        
        super(CuboidODBAModule, self).__init__() 
        
        self._max_train_iter = total_num_steps
        
        if oc_file is not None:
            oc_from_file = OmegaConf.load(open(oc_file, "r"))
        else:
            oc_from_file = None
            
        oc = self.get_base_config(oc_from_file=oc_from_file) # see line 184 for get_base_config method
        model_cfg = OmegaConf.to_object(oc.model)
        num_blocks = len(model_cfg["enc_depth"])
        
        if isinstance(model_cfg["self_pattern"], str):
            enc_attn_patterns = [model_cfg["self_pattern"]] * num_blocks
        else:
            enc_attn_patterns = OmegaConf.to_container(model_cfg["self_pattern"])
            
        if isinstance(model_cfg["cross_self_pattern"], str):
            dec_self_attn_patterns = [model_cfg["cross_self_pattern"]] * num_blocks
        else:
            dec_self_attn_patterns = OmegaConf.to_container(model_cfg["cross_self_pattern"])
            
        if isinstance(model_cfg["cross_pattern"], str):
            dec_cross_attn_patterns = [model_cfg["cross_pattern"]] * num_blocks
        else:
            dec_cross_attn_patterns = OmegaConf.to_container(model_cfg["cross_pattern"])
            
# set the transformer model within the initialization method for the self object
        self.torch_nn_module = CuboidTransformerModel(
            input_shape=model_cfg["input_shape"],
            target_shape=model_cfg["target_shape"],
            base_units=model_cfg["base_units"],
            block_units=model_cfg["block_units"],
            scale_alpha=model_cfg["scale_alpha"],
            enc_depth=model_cfg["enc_depth"],
            dec_depth=model_cfg["dec_depth"],
            enc_use_inter_ffn=model_cfg["enc_use_inter_ffn"],
            dec_use_inter_ffn=model_cfg["dec_use_inter_ffn"],
            dec_hierarchical_pos_embed=model_cfg["dec_hierarchical_pos_embed"],
            downsample=model_cfg["downsample"],
            downsample_type=model_cfg["downsample_type"],
            enc_attn_patterns=enc_attn_patterns,
            dec_self_attn_patterns=dec_self_attn_patterns,
            dec_cross_attn_patterns=dec_cross_attn_patterns,
            dec_cross_last_n_frames=model_cfg["dec_cross_last_n_frames"],
            dec_use_first_self_attn=model_cfg["dec_use_first_self_attn"],
            num_heads=model_cfg["num_heads"],
            attn_drop=model_cfg["attn_drop"],
            proj_drop=model_cfg["proj_drop"],
            ffn_drop=model_cfg["ffn_drop"],
            upsample_type=model_cfg["upsample_type"],
            ffn_activation=model_cfg["ffn_activation"],
            gated_ffn=model_cfg["gated_ffn"],
            norm_layer=model_cfg["norm_layer"],
            
            # global vectors
            num_global_vectors=model_cfg["num_global_vectors"],
            use_dec_self_global=model_cfg["use_dec_self_global"],
            dec_self_update_global=model_cfg["dec_self_update_global"],
            use_dec_cross_global=model_cfg["use_dec_cross_global"],
            use_global_vector_ffn=model_cfg["use_global_vector_ffn"],
            use_global_self_attn=model_cfg["use_global_self_attn"],
            separate_global_qkv=model_cfg["separate_global_qkv"],
            global_dim_ratio=model_cfg["global_dim_ratio"],
            
            # initial_downsample
            initial_downsample_type=model_cfg["initial_downsample_type"],
            initial_downsample_activation=model_cfg["initial_downsample_activation"],
            
            # initial_downsample_type=="conv"
            initial_downsample_scale=model_cfg["initial_downsample_scale"],
            initial_downsample_conv_layers=model_cfg["initial_downsample_conv_layers"],
            final_upsample_conv_layers=model_cfg["final_upsample_conv_layers"],
            
            # misc
            padding_type=model_cfg["padding_type"],
            z_init_method=model_cfg["z_init_method"],
            checkpoint_level=model_cfg["checkpoint_level"],
            pos_embed_type=model_cfg["pos_embed_type"],
            use_relative_pos=model_cfg["use_relative_pos"],
            self_attn_use_final_proj=model_cfg["self_attn_use_final_proj"],
            
            # initialization
            attn_linear_init_mode=model_cfg["attn_linear_init_mode"],
            ffn_linear_init_mode=model_cfg["ffn_linear_init_mode"],
            conv_init_mode=model_cfg["conv_init_mode"],
            down_up_linear_init_mode=model_cfg["down_up_linear_init_mode"],
            norm_init_mode=model_cfg["norm_init_mode"],
        )
        
 # set other self caracteritics 
        self.total_num_steps = total_num_steps
        if oc_file is not None:
            oc_from_file = OmegaConf.load(open(oc_file, "r"))
        else:
            oc_from_file = None
        oc = self.get_base_config(oc_from_file=oc_from_file)
        self.save_hyperparameters(oc)
        self.oc = oc
        
        # layout
        self.in_len = oc.layout.in_len
        self.out_len = oc.layout.out_len
        self.layout = oc.layout.layout
        self.augmentation = oc.layout.augmentation 
        
        # optimization
        self.max_epochs = oc.optim.max_epochs
        self.optim_method = oc.optim.method
        self.lr = oc.optim.lr
        self.wd = oc.optim.wd
        
        # lr_scheduler
        self.total_num_steps = total_num_steps
        self.lr_scheduler_mode = oc.optim.lr_scheduler_mode
        self.warmup_percentage = oc.optim.warmup_percentage
        self.min_lr_ratio = oc.optim.min_lr_ratio
        
        # logging
        self.save_dir = save_dir
        self.logging_prefix = oc.logging.logging_prefix
        
        # trainer
        self.precision = oc.trainer.precision
        
        # visualization
        self.train_example_data_idx_list = list(oc.vis.train_example_data_idx_list)
        self.val_example_data_idx_list = list(oc.vis.val_example_data_idx_list)
        self.test_example_data_idx_list = list(oc.vis.test_example_data_idx_list)
        self.eval_example_only = oc.vis.eval_example_only

        self.configure_save(cfg_file_path=oc_file)

        self.valid_mse = torchmetrics.MeanSquaredError()
        self.valid_mae = torchmetrics.MeanAbsoluteError()
        self.valid_ssim = torchmetrics.image.StructuralSimilarityIndexMeasure()
        self.test_mse = torchmetrics.MeanSquaredError()
        self.test_mae = torchmetrics.MeanAbsoluteError()
        self.test_ssim = torchmetrics.image.StructuralSimilarityIndexMeasure()
        
        self.y_sum_epoch = 0
        self.y_hat_sum_epoch = 0
        
        # added to be compatible with on_validation_end change
        self.H = None
        self.W = None
        
        self.test = test
        
        
    def configure_save(self, cfg_file_path=None):
        
        self.save_dir = os.path.join(exps_dir, self.save_dir) # sets save directory 
        os.makedirs(self.save_dir, exist_ok=True) # create save directory if not existant
        
        if cfg_file_path is not None: # copies config file in save directory 
            cfg_file_target_path = os.path.join(self.save_dir, "cfg.yaml")
            if (not os.path.exists(cfg_file_target_path)) or \
                    (not os.path.samefile(cfg_file_path, cfg_file_target_path)):
                copyfile(cfg_file_path, cfg_file_target_path)
                
        self.example_save_dir = os.path.join(self.save_dir, "examples") 
        os.makedirs(self.example_save_dir, exist_ok=True) # sets and creates a directory for examples/visualization
        
        self.example_save_dir_test = os.path.join(self.save_dir, "test_examples") 
        os.makedirs(self.example_save_dir_test, exist_ok=True) # sets and creates a directory for examples/visualization
        
        os.makedirs(os.path.join(self.save_dir, "test_examples", "results"), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "test_examples", "targets"), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "test_examples", "inputs"), exist_ok=True)
        
        
    def get_base_config(self, oc_from_file=None):
        
        oc = OmegaConf.create() #creates base config
        
        # populates the base config with subconfigs 
        oc.layout = self.get_layout_config()
        oc.optim = self.get_optim_config()
        oc.logging = self.get_logging_config()
        oc.trainer = self.get_trainer_config()
        oc.vis = self.get_vis_config()
        oc.model = self.get_model_config()
        
        if oc_from_file is not None: # merges the config with config file if provided 
            oc = OmegaConf.merge(oc, oc_from_file)
            
        return oc
    
    
    @classmethod 
    def get_model_config(cls):
        
        cfg = OmegaConf.create()
        height = 64
        width = 64
        in_len = 10
        out_len = 10
        data_channels = 1
        cfg.input_shape = (in_len, height, width, data_channels)
        cfg.target_shape = (out_len, height, width, data_channels)

        cfg.base_units = 64
        cfg.block_units = None # multiply by 2 when downsampling in each layer
        cfg.scale_alpha = 1.0

        cfg.enc_depth = [1, 1]
        cfg.dec_depth = [1, 1]
        cfg.enc_use_inter_ffn = True
        cfg.dec_use_inter_ffn = True
        cfg.dec_hierarchical_pos_embed = True

        cfg.downsample = 2
        cfg.downsample_type = "patch_merge"
        cfg.upsample_type = "upsample"

        cfg.num_global_vectors = 8
        cfg.use_dec_self_global = True
        cfg.dec_self_update_global = True
        cfg.use_dec_cross_global = True
        cfg.use_global_vector_ffn = True
        cfg.use_global_self_attn = False
        cfg.separate_global_qkv = False
        cfg.global_dim_ratio = 1

        cfg.self_pattern = 'axial'
        cfg.cross_self_pattern = 'axial'
        cfg.cross_pattern = 'cross_1x1'
        cfg.dec_cross_last_n_frames = None

        cfg.attn_drop = 0.1
        cfg.proj_drop = 0.1
        cfg.ffn_drop = 0.1
        cfg.num_heads = 4

        cfg.ffn_activation = 'gelu'
        cfg.gated_ffn = False
        cfg.norm_layer = 'layer_norm'
        cfg.padding_type = 'zeros'
        cfg.pos_embed_type = "t+hw"
        cfg.use_relative_pos = True
        cfg.self_attn_use_final_proj = True
        cfg.dec_use_first_self_attn = False

        cfg.z_init_method = 'zeros'  # The method for initializing the first input of the decoder
        cfg.initial_downsample_type = "conv"
        cfg.initial_downsample_activation = "leaky"
        cfg.initial_downsample_scale = 2
        cfg.initial_downsample_conv_layers = 2
        cfg.final_upsample_conv_layers = 1
        cfg.checkpoint_level = 2
        
        # initialization
        cfg.attn_linear_init_mode = "0"
        cfg.ffn_linear_init_mode = "0"
        cfg.conv_init_mode = "0"
        cfg.down_up_linear_init_mode = "0"
        cfg.norm_init_mode = "0"
        
        return cfg

    @staticmethod
    def get_layout_config():
        
         oc = OmegaConf.create()
         oc.in_len = 10
         oc.out_len = 10
         oc.layout = "NTHWC" # The layout of the data, not the model
         oc.augmentation = False 
         
         return oc
     
        
    @staticmethod
    def get_optim_config():
        
        oc = OmegaConf.create()
        oc.seed = None
        oc.total_batch_size = 32
        oc.micro_batch_size = 8

        oc.method = "adamw"
        oc.lr = 1E-3
        oc.wd = 1E-5
        oc.gradient_clip_val = 1.0
        oc.max_epochs = 50
        
        # scheduler
        oc.warmup_percentage = 0.2
        oc.lr_scheduler_mode = "cosine"  # Can be strings like 'linear', 'cosine', 'platue'
        oc.min_lr_ratio = 0.1
        oc.warmup_min_lr_ratio = 0.1
        
        # early stopping
        oc.early_stop = False
        oc.early_stop_mode = "min"
        oc.early_stop_patience = 5
        oc.save_top_k = 1
        
        return oc
    
    
    @staticmethod
    def get_logging_config():
        
        oc = OmegaConf.create()
        oc.logging_prefix = "ODBA"
        oc.monitor_lr = True
        oc.monitor_device = False
        cfg.use_wandb = False
        
        return oc
        
    @staticmethod
    def get_trainer_config():
        
        oc = OmegaConf.create()
        oc.check_val_every_n_epoch = 1
        oc.log_step_ratio = 0.001  # Logging every 1% of the total training steps per epoch
        oc.precision = 32
        
        return oc
        
    @staticmethod
    def get_vis_config():
        
        oc = OmegaConf.create()
        oc.train_example_data_idx_list = [0, ]
        oc.val_example_data_idx_list = [0, ]
        oc.test_example_data_idx_list = [0, ]
        oc.eval_example_only = False
        
        return oc
    
    def configure_optimizers(self):
        
# disables the weight decay for layer norm weights and all bias terms
        decay_parameters = get_parameter_names(self.torch_nn_module, [nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [{
            'params': [p for n, p in self.torch_nn_module.named_parameters() if n in decay_parameters],
            'weight_decay': self.oc.optim.wd
        }, {
            'params': [p for n, p in self.torch_nn_module.named_parameters() if n not in decay_parameters],
            'weight_decay': 0.0
        }]
            
# configures optimizer
        if self.oc.optim.method == 'adamw':
            optimizer = torch.optim.AdamW(params=optimizer_grouped_parameters,
                                          lr=self.oc.optim.lr,
                                          weight_decay=self.oc.optim.wd)
        else:
            raise NotImplementedError

        warmup_iter = int(np.round(self.oc.optim.warmup_percentage * self.total_num_steps))
        
# set up warmups
        if self.oc.optim.lr_scheduler_mode == 'cosine':
            warmup_scheduler = LambdaLR(optimizer,
                                        lr_lambda=warmup_lambda(warmup_steps=warmup_iter,
                                                                min_lr_ratio=self.oc.optim.warmup_min_lr_ratio))
            cosine_scheduler = CosineAnnealingLR(optimizer,
                                                 T_max=(self.total_num_steps - warmup_iter),
                                                 eta_min=self.oc.optim.min_lr_ratio * self.oc.optim.lr)
            lr_scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
                                        milestones=[warmup_iter])
        
        elif self.oc.optim.lr_scheduler_mode == 'constant':
            
            if hasattr(self.oc, 'checkpoint_lr') and self.oc.checkpoint_lr is not None:
                # Override with the saved learning rate from the checkpoint
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.oc.checkpoint_lr
            else:
                # Use the constant learning rate defined in the config
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.oc.optim.lr
                
            lr_scheduler = StepLR(optimizer, step_size=1e10, gamma=1.0)
        
        
        else:
            raise NotImplementedError(f"LR scheduler mode {self.oc.optim.lr_scheduler_mode} is not supported.")
        
        lr_scheduler_config = {
            'scheduler': lr_scheduler,
            'interval': 'step',
            'frequency': 1,
            }
        
        for param_group in optimizer.param_groups:
            print(f"Learning rate after scheduler setup: {param_group['lr']}")
    
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler_config}
            
    
    def set_trainer_kwargs(self, **kwargs):
        
# allows checkpointing during training 
        checkpoint_callback = ModelCheckpoint(
            monitor="valid_loss_epoch",
            dirpath=os.path.join(self.save_dir, "checkpoints"),
            filename="model-{epoch:03d}",
            save_top_k=self.oc.optim.save_top_k,
            save_last=True,
            mode="min",
        )

# defines callbacks = way to access parameters during the training 
        callbacks = kwargs.pop("callbacks", [])
        assert isinstance(callbacks, list)
        for ele in callbacks:
            assert isinstance(ele, Callback)
        callbacks += [checkpoint_callback, ]
        if self.oc.logging.monitor_lr:
            callbacks += [LearningRateMonitor(logging_interval='step'), ]
        if self.oc.logging.monitor_device:
            callbacks += [DeviceStatsMonitor(), ]
        if self.oc.optim.early_stop:
            callbacks += [EarlyStopping(monitor="valid_loss_epoch",
                                        min_delta=0.05,
                                        patience=self.oc.optim.early_stop_patience,
                                        verbose=False,
                                        mode=self.oc.optim.early_stop_mode), ]
            
# defines logger objects = record and store training info 
        logger = kwargs.pop("logger", [])
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=self.save_dir)
        csv_logger = pl_loggers.CSVLogger(save_dir=self.save_dir)
        logger += [tb_logger, csv_logger]
        if self.oc.logging.use_wandb:
            wandb_logger = pl_loggers.WandbLogger(project=self.oc.logging.logging_prefix,
                                                  save_dir=self.save_dir)
            logger += [wandb_logger, ]
            
# logging frequency 
        log_every_n_steps = max(1, int(self.oc.trainer.log_step_ratio * self.total_num_steps))
        trainer_init_keys = inspect.signature(Trainer).parameters.keys()
        ret = dict(
            callbacks=callbacks,
            # log
            logger=logger,
            log_every_n_steps=log_every_n_steps,
            # save
            default_root_dir=self.save_dir,
            # ddp
            accelerator="gpu",
            strategy="auto",
            #strategy=ApexDDPStrategy(find_unused_parameters=False, delay_allreduce=True),
            # optimization
            max_epochs=self.oc.optim.max_epochs,
            check_val_every_n_epoch=self.oc.trainer.check_val_every_n_epoch,
            gradient_clip_val=self.oc.optim.gradient_clip_val,
            # NVIDIA amp
            precision=self.oc.trainer.precision,
        )
        
# trainer arguments 
        oc_trainer_kwargs = OmegaConf.to_object(self.oc.trainer)
        oc_trainer_kwargs = {key: val for key, val in oc_trainer_kwargs.items() if key in trainer_init_keys}
        ret.update(oc_trainer_kwargs)
        ret.update(kwargs)
        return ret
            
            
    @classmethod
    def get_total_num_steps(
            cls,
            num_samples: int,
            total_batch_size: int,
            epoch: int = None):
        if epoch is None:
            epoch = cls.get_optim_config().max_epochs
        return int(epoch * num_samples / total_batch_size)
    
    
    @staticmethod
    def get_odba_datamodule_train(micro_batch_size: int = 1, augmentation = False):
        dm = ODBADataModule(data_dir = train_dir, batch_size=micro_batch_size, augmentation = augmentation)
        return dm
    
    @staticmethod
    def get_odba_datamodule_test(micro_batch_size: int = 1):
        dm = ODBATestDataModule(data_dir = test_dir, batch_size=micro_batch_size)
        return dm
        
        
    def forward(self, batch):
        in_seq, target_seq = batch['input'], batch['target']
        pred_seq = self.torch_nn_module(in_seq)
        loss = F.mse_loss(pred_seq, target_seq)
        return pred_seq, loss
    
    def on_train_start(self):
        # Access the optimizer (it will be created later in the training loop)
        for param_group in self.trainer.optimizers[0].param_groups:
            if self.oc.optim.lr_scheduler_mode == 'constant':
                param_group['lr'] = self.oc.optim.lr
                print(f"Learning rate overridden to: {self.oc.optim.lr}")
    
    
    def training_step(self, batch, batch_idx):
        x, y = batch['input'], batch['target']

        y_hat, loss = self(batch)
        self.save_vis_step_end(
            batch_idx=batch_idx,
            in_seq=x, target_seq=y,
            pred_seq=y_hat,
            mode="train"
        )
        self.log('train_loss', loss, on_step=True, on_epoch=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch['input'], batch['target']

        B, T_out, H, W, C = y.shape
        self.H = H
        self.W = W
                
        if not self.eval_example_only or batch_idx in self.val_example_data_idx_list:
            y_hat, _ = self(batch)
            
            self.save_vis_step_end(
                batch_idx=batch_idx,
                in_seq=x, target_seq=y,
                pred_seq=y_hat,
                mode="val"
            )
            if self.precision == 16:
                y_hat = y_hat.float()
                
            step_mse = self.valid_mse(y_hat, y) * H * W
            step_mae = self.valid_mae(y_hat, y) * H * W
            
            y_hat_reshaped = rearrange(y_hat, "b t h w c -> (b t) c h w")
            y_reshaped = rearrange(y, "b t h w c -> (b t) c h w")

            step_ssim = self.valid_ssim(y_hat_reshaped, y_reshaped)
            
            y_sum = y.sum(dim=(2, 3)) 
            y_hat_sum = y_hat.sum(dim=(2, 3))
            ratio = y_hat_sum / y_sum
            
            self.y_sum_epoch += y_sum.sum().item()
            self.y_hat_sum_epoch += y_hat_sum.sum().item()
            
            self.log('valid_sum_ratio_step', ratio.mean(), prog_bar=True, on_step=True, on_epoch=False)
            self.log('valid_mse_step', step_mse, prog_bar=True, on_step=True, on_epoch=False)
            self.log('valid_mae_step', step_mae, prog_bar=True, on_step=True, on_epoch=False)
            self.log('valid_ssim_step', step_ssim, prog_bar=True, on_step=True, on_epoch=False)
        return H, W
    
    
    def on_validation_epoch_end(self):
         frame_mse = self.valid_mse.compute() * self.H * self.W
         frame_mae = self.valid_mae.compute() * self.H * self.W
         valid_loss = frame_mse
         epoch_ssim = self.valid_ssim.compute()
         
         if self.y_hat_sum_epoch != 0:
             ratio = self.y_hat_sum_epoch / self.y_sum_epoch
         else:
            ratio = float('inf')

         self.log('valid_loss_epoch', valid_loss, prog_bar=True, on_step=False, on_epoch=True)
         self.log('valid_mse_epoch', frame_mse, prog_bar=True, on_step=False, on_epoch=True)
         self.log('valid_mae_epoch', frame_mae, prog_bar=True, on_step=False, on_epoch=True)
         self.log('valid_ssim_epoch', epoch_ssim, prog_bar=True, on_step=False, on_epoch=True)
         self.log('valid_sum_ratio_epoch', ratio, prog_bar=True, on_step=False, on_epoch=True)

         self.valid_mse.reset()
         self.valid_mae.reset()
         self.valid_ssim.reset()
         
         self.y_sum_epoch = 0
         self.y_hat_sum_epoch = 0
         
         
    def test_step(self, batch, batch_idx):
        x, y = batch['input'], batch['target']
        if 'metadata' in batch:
            filenames = batch['metadata']
        else:
            filenames = None

        B, T_out, H, W, C = y.shape
        self.H = H
        self.W = W
        
        if not self.eval_example_only or batch_idx in self.val_example_data_idx_list:
            y_hat, _ = self(batch)
            
            self.save_vis_step_end(
                batch_idx=batch_idx,
                in_seq=x, target_seq=y,
                pred_seq=y_hat,
                mode="test"
            )
            if self.precision == 16:
                y_hat = y_hat.float()
                
            step_mse = self.test_mse(y_hat, y) * H * W
            step_mae = self.test_mae(y_hat, y) * H * W
            
            y_hat_reshaped = rearrange(y_hat, "b t h w c -> (b t) c h w")
            y_reshaped = rearrange(y, "b t h w c -> (b t) c h w")
    
            step_ssim = self.test_ssim(y_hat_reshaped, y_reshaped)
            
            # Log test metrics per step
            self.log('test_mse_step', step_mse, prog_bar=True, on_step=True, on_epoch=False)
            self.log('test_mae_step', step_mae, prog_bar=True, on_step=True, on_epoch=False)
            self.log('test_ssim_step', step_ssim, prog_bar=True, on_step=True, on_epoch=False)
            
        if self.test:
            
            self.save_vis_test(
                batch_idx = batch_idx,
                in_seq = x, 
                target_seq = y,
                pred_seq = y_hat,
                filenames = filenames
            )
        
        
        return H, W
        
    def on_test_epoch_end(self): 
        frame_mse = self.test_mse.compute() * self.H * self.W
        frame_mae = self.test_mae.compute() * self.H * self.W
        epoch_ssim = self.test_ssim.compute()
        print('\n Test mse :', frame_mse.item())

        self.log('test_mse_epoch', frame_mse, prog_bar=True, on_step=False, on_epoch=True)
        self.log('test_mae_epoch', frame_mae, prog_bar=True, on_step=False, on_epoch=True)
        self.log('test_ssim_epoch', epoch_ssim, prog_bar=True, on_step=False, on_epoch=True)
        self.test_mse.reset()
        self.test_mae.reset()
        self.test_ssim.reset()
        
        
    def save_vis_step_end(
            self,
            batch_idx: int,
            in_seq: torch.Tensor, target_seq: torch.Tensor,
            pred_seq: torch.Tensor,
            mode: str = "train"):
        
        if self.local_rank == 0:
            if mode == "train":
                example_data_idx_list = self.train_example_data_idx_list
            elif mode == "val":
                example_data_idx_list = self.val_example_data_idx_list
            elif mode == "test":
                example_data_idx_list = self.test_example_data_idx_list
            else:
                raise ValueError(f"Wrong mode {mode}! Must be in ['train', 'val', 'test'].")
            if batch_idx in example_data_idx_list:
                
                save_dir=self.example_save_dir,
                save_prefix=f'{mode}_epoch_{self.current_epoch}',
                fig_path = os.path.join(save_dir[0], f'{save_prefix[0]}.png')
                traj_cmap = generate_custom_colormap('Spectral_r')
                odba_cmap = generate_custom_colormap('copper_r')
                
                fig, ax = plt.subplots(nrows = 2, ncols = 3, figsize = (16,8))
                
                in_seq_np = np.array(in_seq.cpu().detach().numpy())
                target_seq_np = np.array(target_seq.cpu().detach().numpy())
                pred_seq_np = np.array(pred_seq.cpu().detach().numpy())
                
                x = in_seq_np[0,0,:,:,0]
                w_u = in_seq_np[0,0,:,:,1]
                w_v = in_seq_np[0,0,:,:,2]
                xt = target_seq_np[0,0,:,:,0]
                xp = pred_seq_np[0,0,:,:,0]
                
                im0 = ax[0,0].imshow(w_u, cmap='coolwarm')
                ax[0,0].set_title('Average Normalized u-Wind Speed Over Trip')
                fig.colorbar(im0, ax=ax[0,0])
                
                im1 = ax[0,1].imshow(w_v, cmap='coolwarm')
                ax[0,1].set_title('Average Normalized v-Wind Speed Over Trip')
                fig.colorbar(im1, ax=ax[0,1])
                
                im2 = ax[1,0].imshow(x, cmap = traj_cmap, vmin = 0)
                ax[1,0].set_title('Input Trajectory')
                fig.colorbar(im2, ax=ax[1,0])
                
                im3 = ax[1,1].imshow(xt, cmap = odba_cmap, vmin = 0)
                ax[1,1].set_title('Real ODBA')
                fig.colorbar(im3, ax=ax[1,1])
                
                im4 = ax[1,2].imshow(xp, cmap = odba_cmap, vmin = 0)
                ax[1,2].set_title('Predicted ODBA')
                fig.colorbar(im4, ax=ax[1,2])
                
                ax[0,2].axis('off')
                
                plt.tight_layout()
                plt.savefig(fig_path)
                plt.close(fig)
                   
                if mode == "val" or mode == 'test' :
                    
                    df_t = pd.DataFrame(xt)
                    csv_path_t = os.path.join(save_dir[0],  f'{save_prefix[0]}_target.csv')
                    df_t.to_csv(csv_path_t)
                    
                    df_p = pd.DataFrame(xp)
                    csv_path_p = os.path.join(save_dir[0], f'{save_prefix[0]}_simu.csv')
                    df_p.to_csv(csv_path_p)
                    
                
    def save_vis_test(
        self,
        batch_idx: int,
        in_seq: torch.Tensor, 
        target_seq: torch.Tensor,
        pred_seq: torch.Tensor,
        filenames: torch.Tensor):
    
        save_dir=self.example_save_dir_test,
        
        for i in range(len(filenames)): 
            save_prefix=f'test_{filenames[i]}',
            fig_path = os.path.join(save_dir[0], f'{save_prefix[0]}.png')
            traj_cmap = generate_custom_colormap('Spectral_r')
            odba_cmap = generate_custom_colormap('copper_r')
            
            fig, ax = plt.subplots(nrows = 2, ncols = 3, figsize = (16,8))
            
            in_seq_np = np.array(in_seq.cpu().detach().numpy())
            target_seq_np = np.array(target_seq.cpu().detach().numpy())
            pred_seq_np = np.array(pred_seq.cpu().detach().numpy())
            
            x = in_seq_np[i,0,:,:,0]
            w_u = in_seq_np[i,0,:,:,1]
            w_v = in_seq_np[i,0,:,:,2]
            xt = target_seq_np[i,0,:,:,0]
            xp = pred_seq_np[i,0,:,:,0]
            
            im0 = ax[0,0].imshow(w_u, cmap='coolwarm')
            ax[0,0].set_title('Average Normalized u-Wind Speed Over Trip')
            fig.colorbar(im0, ax=ax[0,0])
            
            im1 = ax[0,1].imshow(w_v, cmap='coolwarm')
            ax[0,1].set_title('Average Normalized v-Wind Speed Over Trip')
            fig.colorbar(im1, ax=ax[0,1])
            
            im2 = ax[1,0].imshow(x, cmap = traj_cmap, vmin = 0)
            ax[1,0].set_title('Input Trajectory')
            fig.colorbar(im2, ax=ax[1,0])
            
            im3 = ax[1,1].imshow(xt, cmap = odba_cmap, vmin = 0)
            ax[1,1].set_title('Real ODBA')
            fig.colorbar(im3, ax=ax[1,1])
            
            im4 = ax[1,2].imshow(xp, cmap = odba_cmap, vmin = 0)
            ax[1,2].set_title('Predicted ODBA')
            fig.colorbar(im4, ax=ax[1,2])
            
            ax[0,2].axis('off')
                        
            plt.tight_layout()
            plt.savefig(fig_path)
            plt.close(fig)
              
                
            df_t = pd.DataFrame(xt)
            csv_path_t = os.path.join(save_dir[0], 'targets', f'{save_prefix[0]}_target.csv')
            df_t.to_csv(csv_path_t)
            
            df_p = pd.DataFrame(xp)
            csv_path_p = os.path.join(save_dir[0], 'results', f'{save_prefix[0]}_result.csv')
            df_p.to_csv(csv_path_p)
            
            df_i = pd.DataFrame(x)
            csv_path_i = os.path.join(save_dir[0], 'inputs', f'{save_prefix[0]}_input.csv')
            df_i.to_csv(csv_path_i)
                    
         
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', default='tmp_mnist', type=str)
    parser.add_argument('--gpus', default=1, type=int)
    parser.add_argument('--cfg', default=None, type=str)
    parser.add_argument('--test', default = False, type=bool)
    parser.add_argument('--ckpt_name', default=None)
    return parser

def generate_custom_colormap(cmap):
    colormap = plt.colormaps.get_cmap(cmap)  
    newcolors = colormap(np.linspace(0, 1, 256))
    newcolors[0, :] = np.array([1, 1, 1, 1])
    newcmp = mcolors.ListedColormap(newcolors)
    return newcmp
    
def main():
    
    # command line interpretation
    parser = get_parser()
    args = parser.parse_args()  
    
    # arguments loading 
    if args.cfg is not None:
        oc_from_file = OmegaConf.load(open(args.cfg, "r"))
        total_batch_size = oc_from_file.optim.total_batch_size
        micro_batch_size = oc_from_file.optim.micro_batch_size
        max_epochs = oc_from_file.optim.max_epochs
        seed = oc_from_file.optim.seed
        augmentation = oc_from_file.layout.augmentation
    else:
        micro_batch_size = 1
        total_batch_size = int(micro_batch_size * args.gpus)
        max_epochs = None
        seed = 0
        augmentation = False
        
    # sets random seed  
    seed_everything(seed, workers=True)
    
    if args.test == True :
        dm = CuboidODBAModule.get_odba_datamodule_test(micro_batch_size=micro_batch_size)
        dm.setup()
        
        print('Dataset size :', dm.num_train_samples + dm.num_val_samples + dm.num_test_samples)
        
        accumulate_grad_batches = total_batch_size // (micro_batch_size * args.gpus)

        max_epochs = 1

        total_num_steps = CuboidODBAModule.get_total_num_steps(
            epoch=max_epochs,
            num_samples=dm.num_test_samples,
            total_batch_size=total_batch_size
        )
        
        pl_module = CuboidODBAModule(
        total_num_steps=total_num_steps,
        save_dir=args.save,
        oc_file=args.cfg, 
        test = args.test)
                
        trainer_kwargs = pl_module.set_trainer_kwargs(
            devices=args.gpus,
            max_epochs=max_epochs
        )
        
        trainer = Trainer(**trainer_kwargs)
        
        assert args.ckpt_name is not None, "args.ckpt_name is required for test!"
        ckpt_path = os.path.join(pl_module.save_dir, "checkpoints", args.ckpt_name)
        trainer.test(model=pl_module,
                     datamodule=dm,
                     ckpt_path=ckpt_path)
        
    else: 
        dm = CuboidODBAModule.get_odba_datamodule_train(micro_batch_size=micro_batch_size, augmentation=augmentation)
        dm.setup()
        
        print('Dataset size:', dm.num_train_samples + dm.num_val_samples + dm.num_test_samples)
    
        accumulate_grad_batches = total_batch_size // (micro_batch_size * args.gpus)
        
        total_num_steps = CuboidODBAModule.get_total_num_steps(
            epoch=max_epochs,
            num_samples=dm.num_train_samples,
            total_batch_size=total_batch_size
        )
        
        pl_module = CuboidODBAModule(
            total_num_steps=total_num_steps,
            save_dir=args.save,
            oc_file=args.cfg, 
            test=args.test
        )
        
        trainer_kwargs = pl_module.set_trainer_kwargs(
            devices=args.gpus,
            accumulate_grad_batches=accumulate_grad_batches,
        )
        
        trainer = Trainer(**trainer_kwargs)
        print(args.ckpt_name)
        
        if args.ckpt_name is not None:
            ckpt_path = os.path.join(pl_module.save_dir, "checkpoints", args.ckpt_name)
            if not os.path.exists(ckpt_path):
                warnings.warn(f"ckpt {ckpt_path} not exists! Start training from epoch 0.")
                ckpt_path = None
        else:
            ckpt_path = None
    
        # Fit the model using the entire dataset
        trainer.fit(model=pl_module,
                    datamodule=dm,
                    ckpt_path=ckpt_path)
        
        # Test the model
        trainer.test(ckpt_path="best",
                     datamodule=dm)

        
if __name__ == "__main__":
    main()
