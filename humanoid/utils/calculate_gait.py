import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

def get_coefficients(h0, hswing, v0, vswing, hmax, swing_time):
    
    def equations(coeffs):
        a5, a4, a3, a2, a1, a0 = coeffs
        
        # Height at t=0 should be h0
        eq1 = a0 - h0
        
        # Height at t=swing_time should be hswing
        eq2 = a5 * swing_time**5 + a4 * swing_time**4 + a3 * swing_time**3 + a2 * swing_time**2 + a1 * swing_time + a0 - hswing
        
        # Velocity at t=0 should be v0
        eq3 = a1 - v0
        
        # Velocity at t=swing_time should be vswing
        eq4 = 5 * a5 * swing_time**4 + 4 * a4 * swing_time**3 + 3 * a3 * swing_time**2 + 2 * a2 * swing_time + a1 - vswing
        
        # Height at t=swing_time/2 should be hmax
        eq5 = a5 * (swing_time/2)**5 + a4 * (swing_time/2)**4 + a3 * (swing_time/2)**3 + a2 * (swing_time/2)**2 + a1 * (swing_time/2) + a0 - hmax
        
        # Return the deviations from the expected values. These will be minimized by fsolve.
        return (eq1, eq2, eq3, eq4, eq5, a5 + a4 + a3 + a2 + a1 + a0)  
    
    # Solve for the coefficients using the equations above
    return fsolve(equations, (1, 1, 1, 1, 1, 1))


def plot_curves(coeffs, swing_time):
    a5, a4, a3, a2, a1, a0 = coeffs

    def h(t):
        return a5 * t**5 + a4 * t**4 + a3 * t**3 + a2 * t**2 + a1 * t + a0

    def v(t):
        return 5 * a5 * t**4 + 4 * a4 * t**3 + 3 * a3 * t**2 + 2 * a2 * t + a1

    # Define the acceleration function based on the coefficients
    def a(t):
        return 20 * a5 * t**3 + 12 * a4 * t**2 + 6 * a3 * t + 2 * a2

    t_values = np.linspace(0, swing_time, 500)
    h_values = h(t_values)
    v_values = v(t_values)
    a_values = a(t_values)  # Compute acceleration values

    discrete_t_values = np.linspace(0, swing_time, 14)

    plt.figure(figsize=(12, 9))
    
    plt.subplot(3, 1, 1)
    plt.plot(t_values, h_values, label='Height (h(t))')
    plt.scatter(discrete_t_values, h(discrete_t_values), color='black', label='Discrete Height')
    plt.title('Height Curve')
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(t_values, v_values, label='Velocity (v(t))', color='red')
    plt.scatter(discrete_t_values, v(discrete_t_values), color='black', label='Discrete Velocity')
    # print(v(discrete_t_values))
    plt.title('Velocity Curve')
    plt.grid(True)
    plt.legend()

    # Plotting the acceleration curve
    plt.subplot(3, 1, 3)
    plt.plot(t_values, a_values/50, label='Acceleration (a(t))', color='green')
    plt.scatter(discrete_t_values, a(discrete_t_values)/50, color='black', label='Discrete Acceleration')
    # print(a(discrete_t_values)/50)
    plt.title('Acceleration Curve')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

# Set the constraints and swing time
coeffs = get_coefficients(0, 0, 0, -0.1, 0.04, 0.26)

print("Coefficients (a5, a4, a3, a2, a1, a0):")
print(f"a5 = {coeffs[0]:.15f}")
print(f"a4 = {coeffs[1]:.15f}")
print(f"a3 = {coeffs[2]:.15f}")
print(f"a2 = {coeffs[3]:.15f}")
print(f"a1 = {coeffs[4]:.15f}")
print(f"a0 = {coeffs[5]:.15f}")

plot_curves(coeffs, 0.26)