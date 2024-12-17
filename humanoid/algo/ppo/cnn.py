import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, 
                 num_channels_input,
                 num_classes,
                 sequence_length,
                 feature_sizes=[32, 64],  # Example feature sizes for each conv layer
                 kernel_sizes=[5, 5],     # Kernel sizes for each conv layer
                 activation='relu',
                 **kwargs):
        if kwargs:
            print("CNN.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(CNN, self).__init__()

        self.layers = nn.ModuleList()
        self.activation = activation
        
        # Creating 1D convolutional layers
        in_channels = num_channels_input
        for out_channels, kernel_size in zip(feature_sizes, kernel_sizes):
            self.layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size))
            self.layers.append(get_activation(self.activation))
            self.layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
            in_channels = out_channels
        
        # Calculate the size of the features after the conv and pooling layers
        feature_length = sequence_length
        for kernel_size in kernel_sizes:
            feature_length = (feature_length - kernel_size) // 2 + 1
        
        self.num_features_before_fc = in_channels * feature_length
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.num_features_before_fc, 120)  # Example size
        self.fc2 = nn.Linear(120, 84)  # Example size
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        # Convolutional layers
        for layer in self.layers:
            x = layer(x)
        
        # Flatten the output for the fully connected layers
        x = x.view(-1, self.num_features_before_fc)
        
        # Fully connected layers with activation
        x = get_activation(self.activation)(self.fc1(x))
        x = get_activation(self.activation)(self.fc2(x))
        x = self.fc3(x)
        
        return x


# Helper function to get activation function
def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        # Note: CReLU is not directly available in PyTorch, using ReLU instead
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("Invalid activation function!")
        return None