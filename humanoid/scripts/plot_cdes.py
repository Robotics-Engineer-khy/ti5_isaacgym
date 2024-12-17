import numpy as np
import matplotlib.pyplot as plt
import torch

def standard_normal_cdf(x, std):
    return 0.5 * (1 + torch.erf(x / (std * torch.sqrt(torch.tensor(2.0)))))

def c_des_plot():
    phi_stance = 0.5
    num_points = 500
    phi_i = np.linspace(0,1.0,num_points)
    phi_bar = np.linspace(0,1,num_points)

    # if(phi_i <= phi_stance):
    for i in range(500):
        if(phi_i[i:i+1]<= phi_stance):
            phi_bar[i:i+1] = 0.5 * phi_i[i:i+1] / (phi_stance)
        else:
            phi_bar[i:i+1] = 0.5 + 0.5 * (phi_i[i:i+1] - phi_stance) / (1 - phi_stance)

    sigma = 0.02
    C_des = standard_normal_cdf(phi_bar, sigma) * (1 - standard_normal_cdf(phi_bar - 0.5, sigma)) \
            + standard_normal_cdf(phi_bar - 1, sigma) * (1 - standard_normal_cdf(phi_bar - 1.5, sigma))
    fig,ax = plt.subplots()
    ax.plot(phi_i,C_des,color='blue')
    ax.set_xlabel('phi_i')
    ax.set_ylabel('C_des')
    ax.grid(True)
    ax.legend()
    plt.show()
if __name__ == "__main__":
    c_des_plot()

