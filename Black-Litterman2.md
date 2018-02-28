
# Black-Litterman 코드 연습 파일
Black-litterman 모델을 파이썬 코드로 작성한 파일 입니다.


```python
import pandas as pd
import numpy as np
import numpy.linalg as lin
import scipy as spy
from scipy.optimize import minimize
import matplotlib.pyplot as plt
```

## 1. Used data
> 1. data_rets : 20171018-20180117 일일 수익률(종목 : 'AAPL','INTC','AEP','AMZN','MRK','XOM','FB','TSLA') 
> 2. data_cap : 20180118 기준
> 3. tau : 0.025 일반 상수로 사용
> 4. risk_free : 20180118 3-month TB


```python
data_rets = pd.read_excel('D:\\제용\\2017동계 현장실습\\논문\\Black-Litterman\\black_litterman_ret.xlsx', sheetname = 'Sheet1', parse_cols = 'A:ZZ', index_col = 0)
data_cap = pd.read_excel('D:\\제용\\2017동계 현장실습\\논문\\Black-Litterman\\black_litterman_market_cap.xlsx', sheetname = 'Sheet1', parse_cols = 'A:ZZ')
tau = 0.025
risk_free = 0.01445/250
```


```python
market_weight = np.matrix(data_cap/np.sum(data_cap['size']))
sigma = np.cov(data_rets.T)#*250

mean = np.sum(np.dot(np.mean(data_rets), market_weight))
var = np.dot(np.dot(market_weight.T, sigma), market_weight)
```


```python
lmb = (mean - risk_free) / var
```

## lmb : \\(\lambda\\) = \\(\frac{E(r)-r_f}{\sigma^2}\\)
> 위험회피계수


```python
lmb
```




    matrix([[ 20.70030433]])




```python
pi = np.dot(sigma, market_weight) * lmb  
```

## pi : \\(\Pi\\) = \\(\lambda\Sigma(w_{mkt})\\)
> 균형기대수익률


```python
pi
```




    matrix([[ 0.00180961],
            [ 0.00198575],
            [ 0.00012886],
            [ 0.00323418],
            [-0.00102157],
            [ 0.00034535],
            [ 0.00188632],
            [ 0.00123158]])



## 2. \\(Sharp\\)  \\(ratio\\) 최적화 \\(function\\)


```python
def solve_weights_sharp(R, sigma, rf):
    def objective_fun(W, R, sigma, rf):
        mean = np.sum(R * W)
        var = np.dot(np.dot(W, sigma), W)
        sharp_ratio = (mean - rf) / np.sqrt(var)
        return 1/sharp_ratio
    n = len(R)
    W = [np.ones([n]) / n]
    bound = [(0., 1.) for i in range(n)]
    cons = ({'type': 'eq', 'fun': lambda W: sum(W) - 1.0})
    optimized = minimize(objective_fun, W, (R, sigma, rf), method='SLSQP', constraints=cons, bounds=bound)
    if not optimized.success: raise BaseException(optimized.message)
    return optimized.x
```

## 3. 전망 행렬과 시장전망 지정행렬 \\(function\\)


```python
def views_matrix(absol_views=[], relat_views=[]):
    Q_absol = np.matrix([absol_views[i][2] for i in range(len(absol_views))])
    Q_relat = np.matrix([relat_views[i][3] for i in range(len(relat_views))])
    Q = np.concatenate((Q_absol, Q_relat), axis = 1)
    return Q.T
```


```python
def P_matrix(data_rets, absol_views=[], relat_views=[]):
    P = np.zeros([len(relat_views)+len(absol_views), len(data_rets.columns)])
    stock_name = dict()  
    for i, stock_symbol in enumerate(data_rets.columns.values):
        stock_name[stock_symbol] = i
    for i, view_number in enumerate(absol_views):
        symbol = absol_views[i][0]
        P[i, stock_name[symbol]] = +1 if absol_views[i][1] == '=' else +0
    for i, view_number in enumerate(relat_views):
        symbol1, symbol2 = relat_views[i][0], relat_views[i][2]
        P[i+len(absol_views), stock_name[symbol1]] = +1 if relat_views[i][1] == '>' else -1
        P[i+len(absol_views), stock_name[symbol2]] = -1 if relat_views[i][1] == '>' else +1
    return P
```

### 3.1 투자자의 관점
> 투자자 관점의 input 값으로 전망행렬을 생성


```python
relat_views = [('INTC', '>', 'AAPL', 0.01, 0.8), ('AMZN', '>', 'FB', 0.2, 0.8), ('MRK', '>', 'TSLA', 0.02, 0.8)]     
```


```python
Q = views_matrix(relat_views = relat_views)
```

## Q : \\(Q\\) = \\(P{\cdot}E[R]+ \varepsilon \\)
> 시장전망하 수익률 행렬


```python
Q
```




    matrix([[ 0.01],
            [ 0.2 ],
            [ 0.02]])




```python
P = P_matrix(data_rets=data_rets, relat_views=relat_views)
```

## P : \\(P\\)
> 시장 전망 지정 행렬

> 상대적 관점의 경우에는 -1, 1의 값을, 절대적 관점의 경우에는 1의 값을 가진다.


```python
P
```




    array([[-1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  1.,  0.,  0., -1.,  0.],
           [ 0.,  0.,  0.,  0.,  1.,  0.,  0., -1.]])




```python
def omega_matrix(P, sigma, tau):
    omega = np.dot(np.dot(np.dot(tau, P), sigma), np.transpose(P))
    for i in range(len(P)):
        for j in range(len(P)):
            omega[i,j] = 0 if i != j else omega[i, j]
    return omega
```


```python
omega = omega_matrix(P, sigma, tau)
```

## omega : \\(\varOmega\\) = \\(\pmatrix{{\sigma_1}&{\ldots}&{0} \cr {\vdots}&{\ddots}&{\vdots} \cr {0}&{\ldots}&{\sigma_n}}\\) = \\(\pmatrix{{p_1}\Sigma{p_1}{\tau}&{\ldots}&{0} \cr {\vdots}&{\ddots}&{\vdots} \cr {0}&{\ldots}&{p_n}\Sigma{p_n}{\tau}}\\)
> 시장 전망하에서의 오차항의 공분산 행렬


```python
omega
```




    array([[  6.12183074e-06,   0.00000000e+00,   0.00000000e+00],
           [  0.00000000e+00,   6.34058788e-06,   0.00000000e+00],
           [  0.00000000e+00,   0.00000000e+00,   1.56228371e-05]])



## 2. cofidence matrix
> 전망의 신뢰수준에 따른 결합기대수익률의 변화를 위한 \\(function\\) 알고리즘 작성중...


```python
def confidence_matrix(data_rets, absol_views=[], relat_views=[]):
    confidence_matrix_list =[]
    stock_name = dict()
    for i, stock_symbol in enumerate(data_rets.columns.values):
        stock_name[stock_symbol] = i
    for i, view_number in enumerate(absol_views):
        C_k = np.zeros([len(data_rets.columns), 1])    
        symbol = absol_views[i][0]
        C_k[stock_name[symbol], 0] = +0 if absol_views[i][1] == '!' else +float(absol_views[i][3])
        confidence_matrix_list.append(C_k)
    for i, view_number in enumerate(relat_views):
        C_k = np.zeros([len(data_rets.columns), 1])
        symbol1, symbol2 = relat_views[i][0], relat_views[i][2]
        C_k[stock_name[symbol1], 0] = +0 if relat_views[i][1] == '!' else +float(relat_views[i][4])
        C_k[stock_name[symbol2], 0] = +0 if relat_views[i][1] == '!' else +float(relat_views[i][4])
        confidence_matrix_list.append(C_k)
    return confidence_matrix_list

C_matrix = confidence_matrix(data_rets=data_rets, relat_views=relat_views)
```


```python
def confidence_views(pi, Q, P, sigma, tau, lmb, market_weight, C_matrix):
    for i in range(len(P)):
        return_k_100 = pi + np.dot(np.dot(np.dot((sigma*tau), np.array([P[i,:]]).T), lin.inv(np.dot(np.dot(np.array([P[i,:]]), sigma*tau), np.array([P[i,:]]).T))), (Q[i,:] - np.dot(np.array([P[i,:]]),pi)))
        w_k_100 = np.dot(lin.inv(sigma*float(lmb)), return_k_100)
        D_k_100 = w_k_100 - market_weight
        Tilt_k = np.array(D_k_100) * np.array(C_matrix[i])
        w_k_percent = market_weight + Tilt_k
        
confidence_matrix = confidence_views(pi, Q, P, sigma, tau, lmb, market_weight, C_matrix)
```


```python
pi_adj = np.dot(lin.inv(lin.inv(sigma*tau) + np.dot(np.dot(np.transpose(P), lin.inv(omega)), P)), (np.dot(lin.inv(sigma*tau), pi) + np.dot(np.dot(np.transpose(P), lin.inv(omega)), Q)))
```

## pi_adj : \\(E(R)\\)
> 전망결합기대수익률


```python
pi_adj
```




    matrix([[ 0.02290435],
            [ 0.03243438],
            [-0.00644803],
            [ 0.08873144],
            [-0.01574953],
            [ 0.00158304],
            [-0.01194537],
            [-0.02426458]])




```python
opt_weight = solve_weights_sharp(pi_adj, sigma, risk_free)
```


```python
opt_weight
```




    array([  5.64294250e-02,   1.33085817e-17,   2.79521770e-01,
             3.65729889e-02,   1.48905693e-01,   4.08293906e-01,
             7.02762164e-02,   0.00000000e+00])




```python
portfoilo_return = np.sum(pi_adj * opt_weight)
```


```python
portfoilo_return
```




    0.087245703527815077




```python
portfoilo_risk = np.dot(np.dot(opt_weight, sigma), opt_weight)
```


```python
portfoilo_risk
```




    1.7654912420153494e-05




```python

```


```python

```
