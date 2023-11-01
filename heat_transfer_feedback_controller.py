"""@author: robin leuering"""
import matplotlib.pyplot as plt
import numpy as np
import control as ct

def style_plot(ax, xlabel='', ylabel='', fontsize=20):
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=fontsize)

# Initialize transfer functions
T1, T2, K_gain = 1, 1, 1
G_S1 = ct.tf(K_gain, [T1, 1])
G_S2 = ct.tf(1, [T2, 1])
G_O1 = G_S1 * G_S2

# Compute step response and system dynamics
desired_poles = np.array([(-1 / (0.1*T1) + 0j), (-1 / (0.2*T2) - 0j)])
sys_ss = ct.tf2ss(G_O1)
A, B, C, D = sys_ss.A, sys_ss.B, sys_ss.C, sys_ss.D
R = ct.acker(A, B, desired_poles)
sys_sp = ct.ss(A - B.dot(R), B, C, D)
G_O2 = ct.ss2tf(sys_sp)
G_O2 = G_O2 * ct.tf(ct.dcgain(G_O1) / ct.dcgain(G_O2), [1])

# Compute responses
t = np.arange(0, ct.step_response(G_O1)[0][-1], 0.01)
y1 = ct.step_response(G_O1, t)[1]
y2 = ct.step_response(G_O2, t)[1]

# Plotting
label_size = 20

# Bode plots
plt.figure()
ct.bode(G_O1, c='#284b64', linewidth=4, label='original_system')
ct.bode(G_O2, c='#3C6E71', linewidth=4, label='tuned_system')

# Style bode plots
style_plot(ax=plt.gcf().axes[0], ylabel='Magnitude (dB)',fontsize=label_size)
style_plot(ax=plt.gcf().axes[1], xlabel='Frequency (rad/sec)', ylabel='Phase (degrees)',fontsize=label_size)
plt.legend(fontsize=label_size)

# Step response plots
plt.figure()
plt.plot(t, y1, c='#284b64', linewidth=4, label='original_system')
plt.plot(t, y2, c='#3C6E71', linewidth=4, label='tuned_system')
style_plot(plt.gcf().axes[0], 'Time [s]', '', label_size)
plt.legend(fontsize=label_size)
plt.title('Step Response', fontsize=label_size)
plt.grid(True)
plt.show()
