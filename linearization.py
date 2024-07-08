import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.integrate import quad

import types
from copy import deepcopy

matplotlib.use('TkAgg')

# Given a function, bounds, and a partition number, 
# returns the piecewise linear approximation of the function at x
def linear_eval(f, x, lims, N):
    if x < lims[0] or x > lims[1]:
        raise ValueError("x must be within the limits")
    
    delta = (lims[1] - lims[0])/N
    
    x0 = (x-lims[0])//delta
    x1 = x0 + 1

    xl = lims[0] + x0*delta
    xr = lims[0] + x1*delta

    return f(xl) + (f(xr) - f(xl))/(xr - xl)*(x - xl)


# Given a (x,f(x)) of a function, returns the linear 
# approximation of the function at x
def linear_eval2(f, x):
    i = 0
    xs, ys = f

    if x < xs[0] or x > xs[-1]:
        raise ValueError("x must be within the limits")
    
    while xs[i] < x:
        i += 1
    return ys[i-1] + (ys[i] - ys[i-1])/(xs[i] - xs[i-1])*(x - xs[i-1])

# Given a function, bounds, and a partition number,
# returns the piecewise constant approximation of the function at x
def constant_eval(u, x, lims, N):
    if x < lims[0] or x >= lims[1]:
        return 0
    
    delta = (lims[1] - lims[0])/N
    
    x0 = (x-lims[0])//delta
    x1 = x0 + 1

    xl = lims[0] + x0*delta
    xr = lims[0] + x1*delta

    return 1/delta * quad(u, xl, xr)[0]
    
# Given a function, bounds, and a partition number,
# returns the piecewise linear approximation of the function
# at each partition point
def pointwise_linear_evaluation(f, lims, N):
    x = np.linspace(lims[0], lims[1], N+1)
    y = f(x)
    return x, y

# Given a function, bounds, and a partition number,
# returns the piecewise constant approximation of the function
# at each partition point. Note that y is one element shorter
# than x
def pointwise_constant_evaluation(u, lims, N):
    x = np.linspace(lims[0], lims[1], N+1)
    x_mids = (x[:-1] + x[1:])/2
    y = np.array([constant_eval(u, x_, lims, N) for x_ in x_mids])
    return x, y

def constant_linspace(u_const):
    x, y = u_const
    x_linspace = []
    y_linspace = []
    for i in range(len(x)-1):
        x_linspace.append(x[i])
        y_linspace.append(y[i])
        x_linspace.append(x[i+1])
        y_linspace.append(y[i])
    return x_linspace, y_linspace

# Given a function--either a function or a tuple of (x,f(x))--and bounds,
# returns the cutoff of the function to the bounds
def function_translation(f, lims, N=None):
    if isinstance(f, types.FunctionType):
        if N == None:
            raise ValueError("if f is a function and not a tuple, N must be specified")
        x, y = pointwise_linear_evaluation(f, lims, N)
    elif isinstance(f, tuple):
        if lims[0] < f[0][0] or lims[1] > f[0][-1]:
            raise ValueError("lims must be within the bounds of f")
        x, y = deepcopy(f)
        for i in range(len(x)):
            if x[i] > lims[0]:
                x = x[i-1:]
                y = y[i-1:]
                break
        for i in range(len(x)-1, -1, -1):
            if x[i] < lims[1]:
                x = x[:i+2]
                y = y[:i+2]
                break

        x[0] = lims[0]
        y[0] = linear_eval2(f, lims[0])
        x[-1] = lims[1]
        y[-1] = linear_eval2(f, lims[1])

    return x, y

def index_of(x, xs):
    if x < xs[0] or x > xs[-1]:
        return None
    
    i = 0
    while True:
        if xs[i] <= x and x <= xs[i+1]:
            return i
        i += 1

# convex hull of a function
def convex_hull(f, lims, N):
    x, y = function_translation(f, lims, N)
    convex_points = []

    i = 0
    while i < len(x):
        if i == 0:
            convex_points.append(i)
            i += 1
        else:
            rad_from_hori = np.argmin(np.arctan2(y[i:] - y[convex_points[-1]], x[i:] - x[convex_points[-1]]))
            convex_points.append(i + rad_from_hori)
            i += rad_from_hori + 1

    return x[convex_points], y[convex_points]

# concave hull of a function
def concave_hull(f, lims, N):
    x, y = function_translation(f, lims, N)    
    concave_points = []

    i = 0
    while i < len(x):
        if i == 0:
            concave_points.append(i)
            i += 1
        else:
            rad_from_hori = np.argmax(np.arctan2(y[i:] - y[concave_points[-1]], x[i:] - x[concave_points[-1]]))
            concave_points.append(i + rad_from_hori)
            i += rad_from_hori + 1
        
    return x[concave_points], y[concave_points]
    

def f(x):
    return -.5*x**4-x**3+6*x**2
    # return x**2/2
    return x**2/2


"""
fxlims = (-2,3)
p = pointwise_linear_evaluation(f, fxlims, 100)
# pp = function_translation(p, [2.4375, 8.0625], 100)
plt.plot(*p,"b-")
# plt.plot(*pp,"-")
# print(concave_hull(f, [-4, 3], 20))
c = convex_hull(p, fxlims, 100)
plt.plot(*c, 'r-')
plt.show()
"""
