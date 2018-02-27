

```python
import pandas as pd
import numpy as np
import numpy.linalg as lin
import scipy as spy
from scipy.optimize import minimize
import matplotlib.pyplot as plt
```


```python
data_rets = pd.read_excel('D:\\제용\\2017동계 현장실습\\논문\\Black-Litterman\\black_litterman_ret.xlsx', sheetname = 'Sheet1', parse_cols = 'A:ZZ', index_col = 0)
data_cap = pd.read_excel('D:\\제용\\2017동계 현장실습\\논문\\Black-Litterman\\black_litterman_market_cap.xlsx', sheetname = 'Sheet1', parse_cols = 'A:ZZ')
tau = 0.025
risk_free = 0.01445/250
```


```python
data_rets, data_cap
```




    (                AAPL      INTC       AEP      AMZN       MRK       XOM  \
     Date                                                                     
     2017-10-18 -0.004425  0.011561 -0.002596 -0.012020  0.004587 -0.002411   
     2017-10-19 -0.023660 -0.003975  0.011231 -0.010421  0.003779 -0.000242   
     2017-10-20  0.001731  0.008481  0.002032 -0.003750  0.002039  0.004472   
     2017-10-23 -0.000512  0.009894 -0.000811 -0.016899 -0.007514  0.001564   
     2017-10-24  0.005955  0.002939  0.000676  0.009935 -0.004574  0.002763   
     2017-10-25 -0.004392 -0.004151 -0.005272 -0.003064 -0.010458 -0.003594   
     2017-10-26  0.006393  0.013977  0.002446 -0.000493 -0.007366  0.003607   
     2017-10-27  0.035830  0.073761  0.003525  0.132164 -0.060494  0.002875   
     2017-10-30  0.022508 -0.000676  0.002027  0.008992 -0.060611 -0.002031   
     2017-10-31  0.013916  0.025242  0.003236 -0.005014  0.006946 -0.002274   
     2017-11-01 -0.012719  0.026819 -0.005376 -0.001448  0.004538  0.006239   
     2017-11-02  0.007310  0.008349 -0.003918 -0.008571  0.000542 -0.004054   
     2017-11-03  0.026114 -0.016136  0.004883  0.015883  0.012462 -0.004190   
     2017-11-06  0.010145  0.013741 -0.004725  0.008150 -0.003211  0.006853   
     2017-11-07  0.003214  0.001713  0.019802  0.002240 -0.005190 -0.002030   
     2017-11-08  0.008180 -0.001710  0.004389  0.008645  0.017809 -0.001316   
     2017-11-09 -0.002043 -0.008565  0.007877 -0.003310 -0.006539  0.005990   
     2017-11-10 -0.003310 -0.015551 -0.009405 -0.003348 -0.012987 -0.003125   
     2017-11-13 -0.004008  0.003730  0.018053  0.003395 -0.006849 -0.000603   
     2017-11-14 -0.015118  0.002404  0.016551  0.006793 -0.001996 -0.007842   
     2017-11-15 -0.013190 -0.008722 -0.010337 -0.008928 -0.003455 -0.012524   
     2017-11-16  0.011947  0.004180  0.002220  0.009408  0.006752 -0.008004   
     2017-11-17 -0.005552 -0.022344 -0.004951 -0.006516  0.000544 -0.003972   
     2017-11-20 -0.000999 -0.000224  0.000786 -0.003160 -0.019928  0.003863   
     2017-11-21  0.018590  0.007172  0.004317  0.011702  0.003142  0.003973   
     2017-11-22  0.010512 -0.006453 -0.002345  0.014629  0.001843  0.002844   
     2017-11-24  0.000057  0.002240 -0.001567  0.025810 -0.000368  0.003946   
     2017-11-27 -0.005029 -0.005810  0.003531  0.008288  0.003680 -0.003807   
     2017-11-28 -0.005859  0.005394  0.003649 -0.001865  0.006966  0.006904   
     2017-11-29 -0.020743 -0.017438  0.001558 -0.027086  0.007828  0.007347   
     ...              ...       ...       ...       ...       ...       ...   
     2017-12-04 -0.007308 -0.004252  0.001942 -0.024433  0.006265  0.001318   
     2017-12-05 -0.000942 -0.023601 -0.014472  0.006720 -0.008004 -0.008137   
     2017-12-06 -0.003714  0.000230  0.006031  0.009443 -0.025462 -0.007359   
     2017-12-07  0.001834 -0.008516 -0.001564  0.006456  0.007728  0.003282   
     2017-12-08  0.000295  0.006267  0.004177  0.001905  0.014607  0.001333   
     2017-12-11  0.019484  0.007151  0.001950  0.005955  0.013856  0.004476   
     2017-12-12 -0.005618 -0.007558 -0.013752 -0.003285  0.012957 -0.003252   
     2017-12-13  0.003320  0.000231  0.006446 -0.000815 -0.002804  0.004350   
     2017-12-14 -0.000290 -0.001846 -0.002875  0.008702 -0.007443 -0.002647   
     2017-12-15  0.010161  0.030051  0.003277  0.004156  0.004106  0.001568   
     2017-12-18  0.014083  0.038151 -0.013588  0.009702 -0.000356 -0.001084   
     2017-12-19 -0.010656  0.016861 -0.013113 -0.002688  0.000178 -0.006028   
     2017-12-20 -0.001089  0.011054 -0.000134 -0.008220 -0.002134  0.005216   
     2017-12-21  0.003785 -0.016821 -0.010201 -0.002429  0.008733  0.011826   
     2017-12-22  0.000000 -0.001283 -0.001220 -0.005448 -0.004240  0.001431   
     2017-12-26 -0.025370 -0.013276 -0.008961  0.007190 -0.000355  0.000119   
     2017-12-27  0.000176  0.000651  0.003425  0.004674  0.000000 -0.000953   
     2017-12-28  0.002814  0.002386  0.005598  0.003248  0.004615  0.001430   
     2017-12-29 -0.010814 -0.001298 -0.001086 -0.014021 -0.005830 -0.004523   
     2018-01-02  0.017905  0.014948 -0.015767  0.016708 -0.000889  0.016619   
     2018-01-03 -0.000174 -0.033938 -0.008424  0.012775 -0.001423  0.019640   
     2018-01-04  0.004645 -0.018338 -0.011839  0.004476  0.016209  0.001384   
     2018-01-05  0.011385  0.006977 -0.002114  0.016163 -0.001052 -0.000806   
     2018-01-08 -0.003714  0.000000  0.008757  0.014425 -0.005791  0.004496   
     2018-01-09 -0.000115 -0.025034 -0.011761  0.004676  0.002471 -0.004246   
     2018-01-10 -0.000230 -0.025676 -0.015302  0.001301  0.008803 -0.007952   
     2018-01-11  0.005680  0.021412 -0.011079  0.017818  0.005236  0.009875   
     2018-01-12  0.010326 -0.003916 -0.018478  0.022339  0.018403  0.006787   
     2018-01-16 -0.005082 -0.002313  0.000593 -0.000260  0.058132 -0.006284   
     2018-01-17  0.016516  0.028975  0.011111 -0.007556 -0.000644  0.011843   
     
                       FB      TSLA  
     Date                            
     2017-10-18 -0.000454  0.010963  
     2017-10-19 -0.008351 -0.021799  
     2017-10-20  0.002406 -0.019073  
     2017-10-23 -0.021202 -0.023414  
     2017-10-24  0.003095  0.000950  
     2017-10-25 -0.006985 -0.034090  
     2017-10-26  0.000176  0.001013  
     2017-10-27  0.042490 -0.016249  
     2017-10-30  0.011187 -0.002462  
     2017-10-31  0.001056  0.035772  
     2017-11-01  0.014440 -0.031521  
     2017-11-02 -0.020475 -0.067958  
     2017-11-03  0.000000  0.022823  
     2017-11-06  0.006986 -0.010814  
     2017-11-07  0.000444  0.010800  
     2017-11-08 -0.003828 -0.005424  
     2017-11-09 -0.001448 -0.004599  
     2017-11-10 -0.004685  0.000000  
     2017-11-13  0.001737  0.040958  
     2017-11-14 -0.003916 -0.021243  
     2017-11-15 -0.000674  0.008422  
     2017-11-16  0.009216  0.003855  
     2017-11-17 -0.003285  0.008160  
     2017-11-20 -0.001452 -0.020029  
     2017-11-21  0.017455  0.029377  
     2017-11-22 -0.005444 -0.016393  
     2017-11-24  0.010560  0.009437  
     2017-11-27  0.001368  0.003993  
     2017-11-28 -0.003333  0.002336  
     2017-11-29 -0.039963 -0.031523  
     ...              ...       ...  
     2017-12-04 -0.020731 -0.004339  
     2017-12-05  0.007931 -0.004915  
     2017-12-06  0.018689  0.031478  
     2017-12-07  0.023174 -0.006448  
     2017-12-08 -0.006328  0.012498  
     2017-12-11  0.000223  0.043728  
     2017-12-12 -0.011617  0.036849  
     2017-12-13  0.007572 -0.005865  
     2017-12-14  0.000505 -0.003362  
     2017-12-15  0.010034  0.016455  
     2017-12-18  0.003552 -0.013335  
     2017-12-19 -0.007245 -0.022929  
     2017-12-20 -0.009025 -0.006403  
     2017-12-21 -0.002473  0.008146  
     2017-12-22 -0.001409 -0.019478  
     2017-12-26 -0.006828 -0.024324  
     2017-12-27  0.009262 -0.017807  
     2017-12-28  0.001689  0.011937  
     2017-12-29 -0.008206 -0.012716  
     2018-01-02  0.028108  0.029484  
     2018-01-03  0.017914 -0.010233  
     2018-01-04 -0.001841 -0.008290  
     2018-01-05  0.013671  0.006230  
     2018-01-08  0.007653  0.062638  
     2018-01-09 -0.002178 -0.008085  
     2018-01-10 -0.000160  0.003326  
     2018-01-11 -0.000373  0.009409  
     2018-01-12 -0.044736 -0.005119  
     2018-01-16 -0.005464  0.011421  
     2018-01-17 -0.004428  0.020879  
     
     [62 rows x 8 columns],          size
     AAPL  911.092
     INTC  207.745
     AEP    33.571
     AMZN  624.024
     MRK   168.997
     XOM   372.866
     FB    516.072
     TSLA   58.346)




```python
market_weight = np.matrix(data_cap/np.sum(data_cap['size']))
sigma = np.cov(data_rets.T)#*250

mean = np.sum(np.dot(np.mean(data_rets), market_weight))
var = np.dot(np.dot(market_weight.T, sigma), market_weight)
```


```python
lmb = (mean - risk_free) / var
```


```python
lmb
```




    matrix([[ 20.70030433]])




```python
pi = np.dot(sigma, market_weight) * lmb  
```


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


```python
relat_views = [('INTC', '>', 'AAPL', 0.01, 0.8), ('AMZN', '>', 'FB', 0.2, 0.8), ('MRK', '>', 'TSLA', 0.02, 0.8)]     
```


```python
Q = views_matrix(relat_views = relat_views)
```


```python
Q
```




    matrix([[ 0.01],
            [ 0.2 ],
            [ 0.02]])




```python
P = P_matrix(data_rets=data_rets, relat_views=relat_views)
```


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


```python
omega
```




    array([[  6.12183074e-06,   0.00000000e+00,   0.00000000e+00],
           [  0.00000000e+00,   6.34058788e-06,   0.00000000e+00],
           [  0.00000000e+00,   0.00000000e+00,   1.56228371e-05]])




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
