class CustomLRScheduler:
    """Custom learning rate scheduler with step-wise decay."""
    
    def __init__(self, optimizer, initial_lr, final_lr, step_epoch=50):
        """
        Initialize the custom learning rate scheduler.
        
        Args:
            optimizer: PyTorch optimizer to schedule
            initial_lr: Learning rate for epochs <= step_epoch
            final_lr: Learning rate for epochs > step_epoch
            step_epoch: Epoch at which to switch from initial_lr to final_lr
        """
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.step_epoch = step_epoch

    def step(self, epoch):
        """Update learning rate based on current epoch."""
        if epoch <= self.step_epoch:
            learning_rate = self.initial_lr 
        else:
            learning_rate = self.final_lr
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate
