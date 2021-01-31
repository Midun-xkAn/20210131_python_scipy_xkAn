import scipy.integrate as integrate
import numpy as np


## 求一重积分
def f(x):
    return x + 1

v, err = integrate.quad(f, 1, 2)  # integrate.quad()求一重积分，err是误差

print(v)
print(err)

from scipy import integrate

# 含参一重积分
def f(x, a, b):
    return a * x + b

result = integrate.quad(f, 1, 2, args=(-1, 1))  # integrate.quad()求一重积分
print(result)


# 有断点的一重积分
def f(x):
    return 1 / np.sqrt(abs(x))

v, err = integrate.quad(f, -1, 1, points=[0])  # integrate.quad()求一重积分
print(v)

## 求二重积分

area = integrate.dblquad(lambda x, y: x*y, 0, 0.5, lambda x: 0, lambda x: 1-2*x) # integrate.dblquad()求二重积分
print(area)

## 求三重积分

f = lambda x, y, z: x
g = lambda x: (1 - x) / 2
h = lambda x, y: (1 - x - 2 * y)

v, err = integrate.tplquad(f, 0, 1, 0, g, 0, h)
print(v)

## 求解一阶常微分方程

# 单个的微分方程

def func(y, t):
    return t * np.sqrt(y)

YS = integrate.odeint(func,y0=1,t=np.arange(0,10.1,0.1)) #integrate.odeint求解标准形式下的一阶常微分方程
YT = integrate.solve_ivp(func,[1,10.1],y0=[1]) #integrate.solve_ivp求解标准形式下的一阶常微分方程

# 微分方程组

def lorenz(w, t, p, r, b):
    # 给出位置矢量w，和三个参数p, r, b计算出
    # dx/dt, dy/dt, dz/dt的值
    x, y, z = w
    # 直接与lorenz的计算公式对应
    return np.array([p*(y-x), x*(r-z)-y, x*y-b*z])

t = np.arange(0, 30, 0.01) # 创建时间点
# 调用ode对lorenz进行求解, 用两个不同的初始值
track1 = integrate.odeint(lorenz, (0.0, 1.00, 0.0), t, args=(10.0, 28.0, 3.0))
track2 = integrate.odeint(lorenz, (0.0, 1.01, 0.0), t, args=(10.0, 28.0, 3.0))




