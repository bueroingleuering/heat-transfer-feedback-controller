"""@author: robin leuering"""
import matplotlib.pyplot as plt
import numpy as np
import control as ct

# Time constants
T1, T2 = 2, 3

# Characteristic polynomial coefficients
a1 = (T1 + T2) / (T1 * T2)
a0 = 1 / (T1 * T2)

# Original state space description
A = np.array([[-1/T1, 0], [1/T2, -1/T2]])
B = np.array([[1/T1], [0]])
C = np.array([[0, 1]])
D = np.array(0)
sys_ss = ct.ss(A, B, C, D)

# Equivalent state space description
A = np.array([[-a1, -a0], [1, 0]])
B = np.array([[a0], [0]])

t1, response = ct.step_response(sys_ss)
t = np.arange(0, t1[-1], 0.01)
w = np.ones(len(t))
_, y1 = ct.step_response(sys_ss, t)

# Solve differential equation
x = [np.array([[0.0], [0.0]])]
for i in range(1, len(t)):
    x_dot = A.dot(x[i - 1]) + B.dot(w[i - 1])
    x.append(x[i - 1] + x_dot * (t[i] - t[i - 1]))
y2 = np.dot(C, np.hstack(x)).transpose()

# Plotting
label_size = 20
plt.figure()
plt.plot(t, y1, c='#284b64', linewidth=8, label='original_system')
plt.plot(t, y2, c='#888888', linewidth=4, linestyle='--', label='equivalent_system')
ax = plt.gcf().axes[0]
ax.tick_params(axis='y', labelsize=label_size)
ax.tick_params(axis='x', labelsize=label_size)
plt.legend(fontsize=label_size)
plt.xlabel('Time [s]', fontsize=label_size)
plt.title('Step Response', fontsize=label_size)
plt.grid(True)
plt.show()
