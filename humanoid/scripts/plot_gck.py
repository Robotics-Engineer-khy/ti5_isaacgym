import torch
import matplotlib.pyplot as plt

# Define the functions using torch
def G_alpha_sigma_torch(x, alpha=1, sigma=1):
    return alpha * torch.exp(-(x / sigma)**2)

def C_alpha_beta_sigma_torch(x, alpha=1, beta=1, sigma=1):
    return alpha * ((x / sigma)**(2 * beta) + 1)**-1

# Define parameters
alpha = 1
sigma = 0.6
beta_values = [1, 20]

# Define the x range using torch
x = torch.linspace(-1, 1, 400) * sigma * 5
print(x.shape)

# Calculate y values for each function using torch
G_values = G_alpha_sigma_torch(x, alpha, sigma)
C_values = {beta: C_alpha_beta_sigma_torch(x, alpha, beta, sigma) for beta in beta_values}

# Plot the functions
plt.figure(figsize=(14, 7))

# Plot G_alpha_sigma_torch
plt.subplot(1, 2, 1)
plt.plot(x.numpy(), G_values.numpy(), label='$G_{\\alpha,\\sigma}(x)$')
plt.title('$G_{\\alpha,\\sigma}(x) = \\alpha \\exp \\left( - \\left( \\frac{x}{\\sigma} \\right)^2 \\right)$')
plt.xlabel('x')
plt.ylabel('$G_{\\alpha,\\sigma}(x)$')
plt.legend()
plt.grid(True)

# Plot C_alpha_beta_sigma_torch for different beta values
plt.subplot(1, 2, 2)
for beta, C_values_beta in C_values.items():
    plt.plot(x.numpy(), C_values_beta.numpy(), label=f'$\\beta = {beta}$')
plt.title('$C_{\\alpha,\\beta,\\sigma}(x) = \\alpha \\left( \\left( \\left( \\frac{x}{\\sigma} \\right)^{2\\beta} + 1 \\right)^{-1} \\right)$')
plt.xlabel('x')
plt.ylabel('$C_{\\alpha,\\beta,\\sigma}(x)$')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

