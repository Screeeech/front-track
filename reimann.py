import numpy as np
import linearization as lin
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

# Given function and left and right states, returns the shock speed
def shock_speed(f, uL, uR):
    return (f(uR) - f(uL))/(uR - uL)

# Given pointwise evaluation of a function (x,f(x)) and left and right states
# returns the shock speed
def shock_speed2(f, uL, uR):
    if uL > uR:
        temp = uL
        uL = uR
        uR = temp

    for i in range(1,len(f[0])):
        if uL <= f[0][i] and uR >= f[0][i-1]:
            return (f[1][i] - f[1][i-1])/(f[0][i] - f[0][i-1])

    
    f_uL = lin.linear_eval2(f, uL)
    f_uR = lin.linear_eval2(f, uR) 
    return (f_uR - f_uL)/(uR - uL)

# Given a function, left and right states, and a partition number for flux, 
# returns the states and shock speeds in between for the reimann problem
def reimann(f, uL, uR, h, N=None):
    if uL == uR:
        return np.array([uL]), np.array([])
    
    if uL > uR:
        f_hull = lin.concave_hull(f, [uR, uL], N=N)
    if uL < uR:
        f_hull = lin.convex_hull(f, [uL, uR], N=N)

    n_waves = int(np.ceil(np.abs(uL-uR)/h))
    u1 = np.minimum(uL, uR)
    u2 = np.maximum(uL, uR)
    
    w = [u1]
    speeds = []
    for j in range(1,n_waves+1):
        w_ = u1 + j/n_waves*(u2 - u1)
        s_ = shock_speed2(f_hull, w[-1], w_)
        
        if len(speeds) != 0:
            if s_ != speeds[-1]:
                w.append(w_)
                speeds.append(s_)
            else:
                w[-1] = w_
        else:
            w.append(w_)
            speeds.append(s_)

    if uR < uL:
        w = w[::-1]
        speeds = speeds[::-1]
    
    return np.array(w), np.array(speeds)

# Given a reimann solution, xlimits, and time, plots the shocks
# propagating through time
def plot_reimann(reimann_sol, xlims, T, N=100, x_offset=0, t_offset=0, show=False):
    w, speeds = reimann_sol
    
    x = np.linspace(xlims[0], xlims[1], N)
    for i in range(len(speeds)):
        m = 1/speeds[i]
        x = np.array([0, speeds[i]*T])
        y = np.array([0, T])
        # plt.plot(x+x_offset, x/speeds[i] + t_offset)
        plt.plot(x+x_offset, y + t_offset)

    if show:
        plt.xlim(xlims)
        plt.ylim([t_offset, T+t_offset])
        plt.show()

def waves_to_const(waves, positions, tol=1e-6):
    u = []
    for i in range(len(waves)):
        for j in range(len(waves[i])):
            if len(u) == 0:
                u.append(waves[i][j])
            else:
                if np.abs(waves[i][j]-u[-1]) > tol:
                    u.append(waves[i][j])
    return np.array(positions), np.array(u)


# r = reimann(lambda x: x**2/2, 1, 0, 0.2, N=10)
# plot_reimann(r, [-1,1], 1, N=10, x_offset=-0.5)
def f(x):
    return x**2/2

"""
uL, uR = (4.750000000000001, 1.75)
fxlims = (0.25,9.25)
M = 100
f_linear = lin.pointwise_linear_evaluation(f, fxlims, M)
plt.plot(*f_linear, "-")
plt.show()
r = reimann(f_linear, uL, uR, 0.2, M)
print(r)
"""
