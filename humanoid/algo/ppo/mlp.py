import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, 
        num_input = 45,
        num_output = 24,
        hidden_dims=[256,128],
        activation='elu',
        **kwargs):
        
        if kwargs:
            print("Dagger.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(MLP, self).__init__()
        
        activation = get_activation(activation)

        #mlp
        mlp_layers = []
        mlp_layers.append(nn.Linear(num_input, hidden_dims[0]))
        mlp_layers.append(activation)
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                mlp_layers.append(nn.Linear(hidden_dims[l], num_output))
            else:
                mlp_layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                mlp_layers.append(activation)
        self.mlp = nn.Sequential(*mlp_layers)

        print(f"MLP: {self.mlp}")
        
    def forward(self,input):
        return self.mlp(input)
    
def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None