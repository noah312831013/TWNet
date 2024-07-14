import torch
import torch.nn as nn
import os

class base_model(nn.Module):

    def __init__(self):
        super().__init__()
        # ======================
        # below is for unit test
        # ======================
        # self.layer1 = nn.Linear(10, 10)  # Example: Linear layer
        # self.layer2 = nn.Linear(10, 10)
        # # Initialize weights and biases
        # nn.init.xavier_uniform_(self.layer1.weight)  # Example initialization
        # nn.init.zeros_(self.layer1.bias)
        # nn.init.xavier_uniform_(self.layer2.weight)
        # nn.init.zeros_(self.layer2.bias)

        self.ckpt_dir = './ckpt'
        self.exist_model_name = None
        self.exist_model_path = None
        self.best_l1loss = float('inf')
        os.makedirs(self.ckpt_dir, exist_ok=True)
        for x in os.listdir(self.ckpt_dir):
            if x.endswith('.pkl'):
                self.exist_model_name = x
                self.exist_model_path = os.path.join(self.ckpt_dir, self.exist_model_name)


    def show_parameter_number(self):

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print('{} parameter total:{:,}, trainable:{:,}'.format(self._get_name(), total, trainable))

    def load(self, device):

        if self.exist_model_name == None:
            print('*'*50)
            print("Empty model folder! Using initial weights")
            print('*'*50)
            return 0

        checkpoint = None
        exist_checkpoint = torch.load(self.exist_model_path, map_location=torch.device(device))
        self.best_l1loss = exist_checkpoint['l1loss']
        self.load_state_dict(exist_checkpoint['model'])
        checkpoint = exist_checkpoint

        if checkpoint is None:
            print("Invalid checkpoint")
            return

        print('*'*50)
        print(f"Load best: {self.exist_model_name}")

        print(f"Best l1loss: {self.best_l1loss}")
        print(f"Last epoch: {checkpoint['epoch'] + 1}")
        print('*'*50)
        return checkpoint['epoch'] + 1
    
    def save(self, epoch, l1loss):

        name = str(l1loss)+'_'+str(epoch)+'.pkl'
        checkpoint = {
            'model': self.state_dict(),
            'epoch': epoch,
            'l1loss': l1loss,
        }

        if l1loss < self.best_l1loss:
            self.best_l1loss = l1loss
            if self.exist_model_path != None:
                os.remove(self.exist_model_path)
            self.exist_model_name = name
            self.exist_model_path = os.path.join(self.ckpt_dir, name)
            torch.save(checkpoint, self.exist_model_path)
            print("#" * 100)
            print(f"Saved best model: {self.exist_model_path}")
            print("#" * 100)