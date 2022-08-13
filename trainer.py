import os
import time
import torch
from loguru import logger
from tqdm import tqdm
from utils import to_cuda
from metrics import AverageMeter,run_online_evaluation
import torch.distributed as dist
import wandb


class Trainer:
    def __init__(self, config, train_loader,val_loader, model,loss,optimizer,lr_scheduler):
        self.config = config
       
        self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        self.loss = loss
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.num_steps = len(self.train_loader)
        if self._get_rank()==0:
            self.checkpoint_dir = os.path.join(config.SAVE_DIR,config.EXPERIMENT_ID)
         
            os.makedirs(self.checkpoint_dir)
    def train(self):
        
        for epoch in range(1, self.config.TRAIN.EPOCHS+1):
            if self.config.DIS:
                self.train_loader.sampler.set_epoch(epoch)
            self._train_epoch(epoch)
            if self.val_loader is not None and epoch % self.config.TRAIN.VAL_NUM_EPOCHS == 0:
                results = self._valid_epoch(epoch)
                if self._get_rank()==0 :
                    logger.info(f'## Info for epoch {epoch} ## ')
                    for k, v in results.items():
                        logger.info(f'{str(k):15s}: {v}')
            if epoch % self.config.TRAIN.VAL_NUM_EPOCHS == 0 and self._get_rank()==0:
                self._save_checkpoint(epoch)
           

    def _train_epoch(self, epoch):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.total_loss = AverageMeter()
        self.DICE = AverageMeter()
    
        self.model.train()

        
        tbar = tqdm(self.train_loader, ncols=150)
        tic = time.time()
        for idx, (data,_) in enumerate(tbar):
            self.data_time.update(time.time() - tic)
            img = to_cuda(data["data"])
            gt = to_cuda(data["seg"])
            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.config.AMP):
                pre = self.model(img)
                loss = self.loss(pre, gt)
            if self.config.AMP:
                self.scaler.scale(loss).backward()
                if self.config.TRAIN.DO_BACKPROP:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 12)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.config.TRAIN.DO_BACKPROP:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 12)
                self.optimizer.step()

            self.total_loss.update(loss.item())
            self.batch_time.update(time.time() - tic)
            self.DICE.update(run_online_evaluation(pre, gt))
            
            tbar.set_description(
                'TRAIN ({}) | Loss: {} | DICE {} |B {} D {} |'.format(
                    epoch, self.total_loss.average, self.DICE.average, self.batch_time.average, self.data_time.average))
            tic = time.time()
       
            self.lr_scheduler.step_update(epoch * self.num_steps + idx)
        if self._get_rank()==0:
            wandb.log({'train/loss': self.total_loss.average,
                    'train/dice': self.DICE.average,
                    'train/lr': self.optimizer.param_groups[0]['lr']},
                      step=epoch)
    def _valid_epoch(self, epoch):
        logger.info('\n###### EVALUATION ######')
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.total_loss = AverageMeter()
        self.DICE = AverageMeter()

        self.model.eval()
    
        tbar = tqdm(self.val_loader, ncols=150)
        tic = time.time()
        with torch.no_grad():

            for idx, (data, _) in enumerate(tbar):
                self.data_time.update(time.time() - tic)
                img = to_cuda(data["data"])
                gt = to_cuda(data["seg"])
                
                with torch.cuda.amp.autocast(enabled=self.config.AMP):
                    
                    pre = self.model(img)
                    loss = self.loss(pre, gt)

                self.total_loss.update(loss.item())
                self.batch_time.update(time.time() - tic)
                
                self.DICE.update(run_online_evaluation(pre, gt))
                tbar.set_description(
                'TEST ({}) | Loss: {} | DICE {} |B {} D {} |'.format(
                    epoch, self.total_loss.average, self.DICE.average, self.batch_time.average, self.data_time.average))
                tic = time.time()
        if self._get_rank()==0:        
            wandb.log({'val/loss': self.total_loss.average,
                      'val/dice': self.DICE.average,
                      'val/batch_time': self.batch_time.average,
                      'val/data_time':  self.data_time.average
                      },
                      step=epoch)
        log = {'val_loss': self.total_loss.average,
               'val_dice': self.DICE.average
        }
        return log
    def _get_rank(self):
        """get gpu id in distribution training."""
        if not dist.is_available():
            return 0
        if not dist.is_initialized():
            return 0
        return dist.get_rank()

    def _save_checkpoint(self, epoch):
        state = {
            'arch': type(self.model).__name__,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config
        }
        filename = os.path.join(self.checkpoint_dir,
                                'final_checkpoint.pth')
        logger.info(f'Saving a checkpoint: {filename} ...')
        torch.save(state, filename)
        return filename

    def _reset_metrics(self):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.total_loss = AverageMeter()
        self.DICE = AverageMeter()
