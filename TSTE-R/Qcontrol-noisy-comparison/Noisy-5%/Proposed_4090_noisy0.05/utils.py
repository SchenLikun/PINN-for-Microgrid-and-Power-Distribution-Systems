import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import csv
import os
from deepxde.callbacks import Callback


class CustomNormalization(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(CustomNormalization, self).__init__()
        self.epsilon = epsilon
        self.lb_main = torch.tensor([0.0, 0.0], dtype=torch.float32)
        self.ub_main = torch.tensor([1.0, 1.0], dtype=torch.float32)

    def forward(self, inputs):
        return 2 * (inputs - self.lb_main) / (self.ub_main - self.lb_main) - 1

class FourierFeaturesLayer(nn.Module):
    def __init__(self, num_features, scale=10.0):
        super(FourierFeaturesLayer, self).__init__()
        self.num_features = num_features
        self.scale = scale
        self.B = None  # Will be initialized in forward

    def forward(self, inputs):
        if self.B is None:
            self.B = nn.Parameter(torch.randn(inputs.shape[-1], self.num_features) * self.scale)
        x_proj = 2.0 * np.pi * torch.matmul(inputs, self.B)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class CustomSineCosineActivation(nn.Module):
    def __init__(self):
        super(CustomSineCosineActivation, self).__init__()
        self.A = nn.Parameter(torch.ones(1))
        self.B = nn.Parameter(torch.ones(1))
        self.C = nn.Parameter(torch.ones(1))
        self.D = nn.Parameter(torch.ones(1))
        self.E = nn.Parameter(torch.ones(1))
        self.F = nn.Parameter(torch.ones(1))

    def forward(self, inputs):
        return self.A * torch.sin(self.B * inputs) + self.C * torch.cos(self.D * inputs) + self.E * torch.tanh(self.F * inputs)

class H_PINN(nn.Module):
    def __init__(self, input_shape=2, output_shape=1, num_fourier_features=64):
        super(H_PINN, self).__init__()
        self.norm1 = CustomNormalization()
        l2_factor = 0.001
        self.input_projection = nn.Linear(input_shape, 100, bias=False)  # Adjust input size if needed
        self.dense1 = nn.Linear(100, 100)
        self.custom_activation1 = CustomSineCosineActivation()
        self.dense2 = nn.Linear(100, 100)
        self.custom_activation2 = CustomSineCosineActivation()
        self.dense3 = nn.Linear(100, 100)
        self.custom_activation3 = CustomSineCosineActivation()
        self.fourier = FourierFeaturesLayer(num_features=num_fourier_features)
        self.output_layer = nn.Linear(100, output_shape)
        self.l2_reg = l2_factor
        self.regularizer = None

    def forward(self, inputs):
        x = self.norm1(inputs)
        projected_x = self.input_projection(x)
        
        x1 = self.dense1(projected_x)
        x1_activated = self.custom_activation1(x1)
        x2_input = x1_activated + projected_x
        
        x2 = self.dense2(x2_input)
        x2_activated = self.custom_activation2(x2)
        x3_input = x2_activated + projected_x
        
        x3 = self.dense3(x3_input)
        x3 = x3 + projected_x
        x3_activated = self.custom_activation3(x3)
        x_out = self.output_layer(x3_activated)
        
        return x_out
    
class EpochTimer(Callback):
    def __init__(self, filepath="timelog.csv"):
        super().__init__()
        self.filepath = filepath
        self.start_time = None
        self.epoch_times = []
        self.epoch_counter = 0
        
        dirpath = os.path.dirname(filepath)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        
        with open(self.filepath, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Time (seconds)"])
    
    def on_train_begin(self):
        self.start_time = time.time()
    
    def on_epoch_end(self):
        self.epoch_counter += 1
        if self.epoch_counter % 1000 == 0:
            elapsed_time = time.time() - self.start_time
            epoch = self.model.train_state.epoch
            self.epoch_times.append((epoch, elapsed_time))

            with open(self.filepath, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([epoch, elapsed_time])
    
    def on_train_end(self):
        print(f"Total training time: {self.epoch_times[-1][1]:.2f} seconds")

