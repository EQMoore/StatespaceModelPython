import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation

dt = 0.1
T = 100
n_steps = int(T / dt)

A = np.array([[0.9, 0.2],
              [-0.1, 0.95]])
B = np.array([[0.1],
              [0.05]])

def simulate(A, B):
    x = np.zeros((2, n_steps))
    x[:, 0] = [1.0, 0.0]
    for k in range(n_steps - 1):
        u = x[0, k]
        x[:, k+1] = A @ x[:, k] + B.flatten() * u
    return x

x = simulate(A, B)
time = np.linspace(0, T, n_steps)

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.1, bottom=0.4)
line1, = ax.plot(time, x[0, :], label='State x1 (input)')
line2, = ax.plot(time, x[1, :], label='State x2')
ax.set_title("State-Space System Response")
ax.set_xlabel("Time [s]")
ax.set_ylabel("State Values")
ax.legend()
ax.grid(True)

axcolor = 'blue'
ax_a11 = plt.axes([0.15, 0.28, 0.65, 0.03], facecolor=axcolor)
ax_a22 = plt.axes([0.15, 0.23, 0.65, 0.03], facecolor=axcolor)
ax_b1  = plt.axes([0.15, 0.18, 0.65, 0.03], facecolor=axcolor)
ax_b2  = plt.axes([0.15, 0.13, 0.65, 0.03], facecolor=axcolor)
ax_button = plt.axes([0.4, 0.05, 0.2, 0.05])

s_a11 = Slider(ax_a11, 'A[0,0]', 0.5, 1.5, valinit=A[0,0])
s_a22 = Slider(ax_a22, 'A[1,1]', 0.5, 1.5, valinit=A[1,1])
s_b1  = Slider(ax_b1,  'B[0]', 0.0, 0.5, valinit=B[0,0])
s_b2  = Slider(ax_b2,  'B[1]', 0.0, 0.5, valinit=B[1,0])
toggle = Button(ax_button, 'Toggle View', color='lightblue', hovercolor='0.8')

view_mode = "time"

fig2, ax2 = plt.subplots()
circle = plt.Circle((0,0), 1, fill=False)
wheel_line, = ax2.plot([0, np.cos(x[0,0])], [0, np.sin(x[0,0])], lw=2)
ax2.add_patch(circle)
ax2.set_aspect('equal', adjustable='box')
ax2.set_xlim(-1.2, 1.2)
ax2.set_ylim(-1.2, 1.2)
eigs = np.linalg.eigvals(A)
eig_text = ax2.text(-1.2, 1.3, f"Eigenvalues:\n{eigs[0].real:+.2f}{eigs[0].imag:+.2f}i\n{eigs[1].real:+.2f}{eigs[1].imag:+.2f}i", fontsize=10)

def update(val):
    A_new = np.array([[s_a11.val, 0.2],
                      [-0.1, s_a22.val]])
    B_new = np.array([[s_b1.val],
                      [s_b2.val]])
    global x
    x = simulate(A_new, B_new)
    eigs_new = np.linalg.eigvals(A_new)
    eig_text.set_text(f"Eigenvalues:\n{eigs_new[0].real:+.2f}{eigs_new[0].imag:+.2f}i\n{eigs_new[1].real:+.2f}{eigs_new[1].imag:+.2f}i")
    if view_mode == "time":
        line1.set_xdata(time)
        line1.set_ydata(x[0, :])
        line2.set_xdata(time)
        line2.set_ydata(x[1, :])
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("State Values")
        ax.set_title("State-Space System Response")
    else:
        line1.set_xdata(x[0, :])
        line1.set_ydata(x[1, :])
        line2.set_xdata([])
        line2.set_ydata([])
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_title("Phase Plot: x2 vs x1")
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw_idle()

def toggle_view(event):
    global view_mode
    view_mode = "phase" if view_mode == "time" else "time"
    update(None)

s_a11.on_changed(update)
s_a22.on_changed(update)
s_b1.on_changed(update)
s_b2.on_changed(update)
toggle.on_clicked(toggle_view)

def animate(i):
    angle = x[0, i % n_steps]
    wheel_line.set_data([0, np.cos(angle)], [0, np.sin(angle)])
    return wheel_line,

ani = FuncAnimation(fig2, animate, frames=n_steps, interval=dt*1000, blit=True)

plt.show()

