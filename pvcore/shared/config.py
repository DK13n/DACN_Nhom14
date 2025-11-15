
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 8
num_frames = 16
frame_size = 224
learning_rate = 1e-4
min_lr = 1e-6
num_epochs = 15
weight_decay = 1e-4
grad_clip = 1.0
use_amp = True
random_state = 312
test_size = 0.2
save_step = 5

use_focal_loss = True
focal_loss_type = "balanced"   # 'balanced' or 'standard'
gamma = 2.0                    # Focusing parameter
beta = 0.999                   # For balanced focal loss
label_smoothing = 0.1          # Label smoothing
use_class_weights = True       # Whether to use class weights
