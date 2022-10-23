from math import pi, sin, cos, exp, prod, fabs, acos
from random import uniform
from sys import stdin, stdout, setrecursionlimit
from gc import disable
import matplotlib.pyplot as plt

gets = input
puts = print
input = stdin.readline
print = stdout.write


N = 7
a, b, alpha, beta = 2.1, 3.3, 0.4, 0

def f(x:float) -> float:
    return 4.5*cos(7*x)*exp(-2*x/3) + 1.4*sin(1.5*x)*exp(-x/3) + 3;

def p(x:float) -> float:
    return 1/(x-a)**alpha

def F(x:float) -> float:
    return f(x)*p(x);
 
def Gauss(A:list, B:list):
    n = len(A)
    a = [[0]*n for i in range(n)]
    for i in range(n):
        for j in range(n):
            a[i][j] = A[i][j]
    [*b] = B
    det = 1
    for i in range(n-1):
        k = i
        for j in range(i+1, n):
            if (fabs(a[j][i]) > fabs(a[k][i])): k = j
        if (k != i):
            a[i], a[k] = a[k], a[i]
            b[i], b[k] = b[k], b[i]
            det = -det
 
        for j in range(i+1, n):
            t = a[j][i]/a[i][i]
            for k in range(i+1, n): a[j][k] -= t*a[i][k]
            b[j] -= t*b[i]
 
    for i in reversed(range(n)):
        for j in range(i + 1, n):
            t = a[i][j]
            b[i] -= t*b[j]
        t = 1/a[i][i]
        det *= a[i][i]
        b[i] *= t
    return det, b

def middle_Riemann_sum(f, down:float, up:float, n:int = 1000) -> float:
    h = (up-down)/n
    return h*sum(f(down + (i - 1/2)*h) for i in range(1,n+1));

def left_Riemann_sum(f, down:float, up:float, n:int = 1000) -> float:
    h = (up-down)/n
    return h*sum(f(down + i*h) for i in range(n));

def trapezoidal_rule(f, down:float, up:float, n:int = 1000) -> float:
    h = (up-down)/n
    return h/2 * (f(down) + f(up) + sum(2*f(down + i*h) for i in range(1,n)));

def composite_Simpson_rule(f, down:float, up:float, n:int = 1000) -> float:
    h = (up-down)/n
    return h/3 * sum(f(down + 2*i*h) + f(down + (2*i-2)*h) + 4*f(down + (2*i-1)*h) for i in range(1,n//2 + 1));

def Durand_Kerner(expr:list) -> list:
    n = 3
    params = [uniform(0,10) for _ in range(n)]   
    pol = lambda x: sum(x**(n-i) * expr[i] for i in range(n+1))
    while (any(abs(pol(params[k])) > 1e-12 for k in range(n))):
        for i in range(n):
            params[i] = params[i] - pol(params[i])/prod((params[i] - params[j]) for j in range(i))/prod((params[i] - params[j]) for j in range(i+1,n))
    return params

def Tartaglia_Kardano(pol:list) -> list:
    b,c,d = pol
    
    q = (2*b*b*b/27 - b*c/3 + d)/2
    p = (3*c - b*b)/9
    r = fabs(p)**0.5
    phi = acos(q/r/r/r)
    
    return -2*r*cos(phi/3) - b/3, 2*r*cos(pi/3 - phi/3) - b/3, 2*r*cos(pi/3 + phi/3) - b/3

def composite_Newton_Cotes_quadrature_rule(f, down:float, up:float, k:int = 1000):
    n = 3
    h = (up - down)/k
    z = [down + i*h for i in range(k+1)]
    
    mu = [[0]*n for i in range(k+1)]
    A = [[0]*n for i in range(k+1)]
    ans = 0
    for i in range(1,k+1):
        mu[i][0] = ((z[i] - down)**(1 - alpha) - (z[i-1] - down)**(1 - alpha))/(1 - alpha)
        mu[i][1] = ((z[i] - down)**(2 - alpha) - (z[i-1] - down)**(2 - alpha))/(2 - alpha) + down*mu[i][0]
        mu[i][2] = ((z[i] - down)**(3 - alpha) - (z[i-1] - down)**(3 - alpha))/(3 - alpha) + 2*down*mu[i][1] - down*down*mu[i][0]
    
        z_m = (z[i-1]+z[i])/2
        A[i][0] = (mu[i][2] - mu[i][1]*(z_m + z[i]) + mu[i][0]*z_m*z[i])/(z_m - z[i-1])/(z[i] - z[i-1])
        A[i][1] = -(mu[i][2] - mu[i][1]*(z[i-1] + z[i]) + mu[i][0]*z[i-1]*z[i])/(z_m - z[i-1])/(z[i] - z_m)
        A[i][2] = (mu[i][2] - mu[i][1]*(z_m + z[i-1]) + mu[i][0]*z_m*z[i-1])/(z[i] - z_m)/(z[i] - z[i-1])
        
        ans += A[i][0]*f(z[i-1]) + A[i][1]*f(z_m) + A[i][2]*f(z[i])
    
    return ans;


def composite_Gaussian_quadrature_rule(f, down:float, up:float, k:int = 100):
    n = 3
    h = (up - down)/k
    ans = 0
    z = [down + i*h for i in range(k+1)]
    mu = [[0]*(2*n) for i in range(k+1)]
    for i in range(1,k+1):
        mu[i][0] = ((z[i] - down)**(1 - alpha) - (z[i-1] - down)**(1 - alpha))/(1 - alpha)
        
        mu[i][1] = ((z[i] - down)**(2 - alpha) - (z[i-1] - down)**(2 - alpha))/(2 - alpha) + down*mu[i][0]
        
        mu[i][2] = ((z[i] - down)**(3 - alpha) - (z[i-1] - down)**(3 - alpha))/(3 - alpha) + 2*down*mu[i][1] - down*down*mu[i][0]
        
        mu[i][3] = ((z[i] - down)**(4 - alpha) - (z[i-1] - down)**(4 - alpha))/(4 - alpha) + 3*down*mu[i][2] - 3*down*down*mu[i][1] + down*down*down*mu[i][0]
        
        mu[i][4] = ((z[i] - down)**(5 - alpha) - (z[i-1] - down)**(5 - alpha))/(5 - alpha) + 4*down*mu[i][3] - 6*down*down*mu[i][2] + 4*down*down*down*mu[i][1] - down**4*mu[i][0]
        
        mu[i][5] = ((z[i] - down)**(6 - alpha) - (z[i-1] - down)**(6 - alpha))/(6 - alpha) + 5*down*mu[i][4] - 10*down*down*mu[i][3] + 10*down*down*down*mu[i][2] - 5*down**4*mu[i][1] + down**5*mu[i][0]
        
        A = [[0]*n for i in range(n)]
        B = [0]*n
        for s in range(n):
            for j in range(n):
                A[s][j] = mu[i][j+s]
            B[s] = -mu[i][n+s]
        det, a_s = Gauss(A,B)
        x_s = Tartaglia_Kardano(reversed(a_s))
        for s in range(n):
            for j in range(n):
                A[s][j] = x_s[j]**s
            B[s] = mu[i][s]
        det, A_s = Gauss(A,B)
        ans += sum(A_s[i]*f(x_s[i]) for i in range(n))
    return ans;


def main() -> int:
    disable()
    f_ans = 2.950730201339454
    F_ans = 4.461512705324278
    
    print("1.1\n")
    print("  Формула средних прямоугольников:\n    Integral f(x) dx, from %.1f to %.1f = %.9f\n"%(a,b,middle_Riemann_sum(f,a,b)))
    print("  Формула левых прямоугольников:\n    Integral f(x) dx, from %.1f to %.1f = %.9f\n"%(a,b,left_Riemann_sum(f,a,b)))
    print("  Формула трапеции:\n    Integral f(x) dx, from %.1f to %.1f = %.9f\n"%(a,b,trapezoidal_rule(f,a,b)))
    print("  Составная формула Симпсона:\n    Integral f(x) dx, from %.1f to %.1f = %.14f\n"%(a,b,composite_Simpson_rule(f,a,b)))
    print("  Символьно:\n    Integral f(x) dx, from %.1f to %.1f = 2.950730201339454...\n"%(a,b))
    
    print("\n1.2\n")
    print("  Составная формула на базе 3-х-точенчной формулы Ньютона-Котеса:\n    Integral f(x)*p(x) dx, from %.1f to %.1f = %.14f\n"%(a,b,composite_Newton_Cotes_quadrature_rule(f,a,b)))
    print("  Составная формула на базе 3-х-точенчной формулы Гаусса:\n    Integral f(x)*p(x) dx, from %.1f to %.1f = %.14f\n"%(a,b,composite_Gaussian_quadrature_rule(f,a,b)))
    print("  Более мощным методом с оценкой 1e-11:\n    Integral f(x)*p(x) dx, from %.1f to %.1f = 4.461512705324278\n"%(a,b))
    
    plt.figure(1)
    plt.title("Формула средних прямоугольников")
    x = list(range(2,101))
    y = [fabs(f_ans - middle_Riemann_sum(f,a,b,x[i])) for i in range(99)]
    plt.xlabel("number of splits")
    plt.ylabel("accuracy")
    plt.ylim(0,0.01)
    plt.grid()
    plt.plot(x,y)
    plt.show()
    
    
    plt.figure(2)
    plt.title("Формула левых прямоугольников")
    x = list(range(2,101))
    y = [fabs(f_ans - left_Riemann_sum(f,a,b,x[i])) for i in range(99)]
    plt.xlabel("number of splits")
    plt.ylabel("accuracy")
    plt.ylim(0,0.01)
    plt.grid()
    plt.plot(x,y)
    plt.show()    
    
    
    plt.figure(3)
    plt.title("Формула трапеции")
    x = list(range(2,101))
    y = [fabs(f_ans - trapezoidal_rule(f,a,b,x[i])) for i in range(99)]
    plt.xlabel("number of splits")
    plt.ylabel("accuracy")
    plt.ylim(0,0.01)
    plt.grid()
    plt.plot(x,y)
    plt.show()    
    
    
    plt.figure(4)
    plt.title("Составная формула Симпсона")
    x = list(range(2,201,2))
    y = [fabs(f_ans - composite_Simpson_rule(f,a,b,x[i])) for i in range(100)]
    plt.xlabel("number of splits")
    plt.ylabel("accuracy")
    plt.ylim(0,0.0001)
    plt.grid()
    plt.plot(x,y)
    plt.show()        
    
    
    plt.figure(5)
    plt.title("Составная формула на базе 3-х-точенчной формулы Ньютона-Котеса")
    x = list(range(2,101))
    y = [fabs(F_ans - composite_Newton_Cotes_quadrature_rule(f,a,b,x[i])) for i in range(99)]
    plt.xlabel("number of splits")
    plt.ylabel("accuracy")
    plt.ylim(0,0.0001)
    plt.grid()
    plt.plot(x,y)
    plt.show()
    
    
    plt.figure(6)
    plt.title("Составная формула на базе 3-х-точенчной формулы Гаусса")
    x = list(range(2,101))
    y = [fabs(F_ans - composite_Gaussian_quadrature_rule(f,a,b,x[i])) for i in range(99)]
    plt.xlabel("number of splits")
    plt.ylabel("accuracy")
    plt.ylim(0,0.0001)
    plt.grid()
    plt.plot(x,y)
    plt.show()
    
    
    return 0;


if (__name__ == "__main__"): 
    main()