import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def example_1D_func(x_interval=[-5,5], h=0.001):
    x = np.arange(x_interval[0], x_interval[1], h)
    y = x ** 2
    return (x,y)

def derivative(x_interval=[-5,5], h=0.001, func=example_1D_func):
    _, y = func(x_interval, h)
    f_prime = np.gradient(y, h)
    return f_prime

x, y = example_1D_func()
der = derivative()

plt.figure(1)
plt.plot(x,y)
plt.axis([np.min(x), np.max(x), np.min(y), np.max(y)])
plt.grid()
plt.xlabel('x')
plt.ylabel('y')

plt.figure(2)
plt.plot(x,-der)
plt.axis([np.min(x), np.max(x), np.min(-der), np.max(-der)])
plt.grid()
plt.xlabel('x')
plt.ylabel('y')

def example_2d_func(x1_interval=[-5,5], x2_interval=[-5,5], h=0.001):
    x1, x2 = np.meshgrid(np.arange(x1_interval[0], x1_interval[1], h), np.arange(x2_interval[0], x2_interval[1], h))
    y = x1 ** 2 + x2 ** 2
    return (y, x1, x2)

def gradient(x1_interval=[-5,5], x2_interval=[-5,5], h=0.001, func=example_2d_func):
    y, _, _ = func(x1_interval, x2_interval, h)
    f_frime_x1, f_prime_x2 = np.gradient(y, h)
    return (f_frime_x1, f_prime_x2)

y, x1, x2 = example_2d_func()
grad_x1, grad_x2 = gradient()

fig = plt.figure(3)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(x1, x2, y, linewidth=1, antialiased=False)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
ax.set_xlim(np.min(x1), np.max(x1))
ax.set_ylim(np.min(x2), np.max(x2))
ax.set_zlim(np.min(y), np.max(y))

plt.figure(4)
plt.quiver(x1[::1000,::1000], x2[::1000,::1000], -grad_x2[::1000,::1000], -grad_x1[::1000,::1000])
plt.grid()
plt.xlabel('x1')
plt.ylabel('x2')
