
# 팩터의 탄생:CAPM과 3팩터 모델
## 삼성전자와 KOSPI daily 수익률(2000-2016) 회귀분석


```python
import pandas as pd
from scipy import stats, polyval, polyfit
from pylab import plot, title, show, legend

data1 = pd.read_excel('C:\\Users\\이한송\\Desktop\\data2.xlsx',index_row=0)
x = data1['코스피'].tolist()
y = data1['삼성전자'].tolist()
```

### 기울기, y절편, r-값, p-값, 추정 오차


```python
slope, intercept, r, p, std = stats.linregress(x,y)
ry = polyval([slope, intercept], x) #추정오차

print(polyfit(x,y,1))
print(slope, intercept, r, p, std)
print(ry)
```

    [1.18196878e+00 4.02030128e-04]
    1.1819687821455458 0.00040203012836770557 0.7693353333620755 0.0 0.015144721607603488
    [ 0.03597929 -0.08079923 -0.03021096 ...  0.00300236 -0.0098811
      0.001584  ]
    

### [그림2.2]삼성전자와 KOSPI 수익률 관계


```python
plot(x,y,'k.') #실제 데이터
plot(x,ry,'r.-') #회귀선
title('samsung & KOSPI (2000~2016)')
legend(['original','regression'])

show()
```


![png](output_5_0.png)


### [표 2.1]삼성전자와 KOSPI 수익률 회귀분석 결과


```python
regression = [intercept, slope, r*r]
df = pd.DataFrame([regression],index=list('값'),columns=['초과 수익률', 'B(MKT)', 'R Square'])
print(df)
```

         초과 수익률    B(MKT)  R Square
    값  0.000402  1.181969  0.591877
    

### [그림2.3] SMB와 HML 누적 수익률(2000~2016)


```python
import numpy as np
import matplotlib.pyplot as plt

data2 = pd.read_excel('C:\\Users\\이한송\\Desktop\\data1.xlsx',index_col=0)
#누적수익률
SMB = np.cumprod(data2['SMB']/100+1)
HML = np.cumprod(data2['HML']/100+1)
date = data2.index
#SMB그래프
fig, ax1 = plt.subplots()
ax1.plot(date, SMB, 'r-')
ax1.set_ylabel('SMB')
#HML그래프
ax2 = ax1.twinx()
ax2.plot(date, HML, 'g-')
ax2.set_ylabel('HML')

title('SMB & HML (2000~2016)')
ax1.legend()
ax2.legend()
plt.show()
```


![png](output_9_0.png)


### [표2.2] 삼성전자와 KOSPI 수익률 회귀분석 결과: CAPM 및 파마-프렌치 모형


```python
import statsmodels.formula.api as sm

data2 = pd.read_excel('C:\\Users\\이한송\\Desktop\\data1.xlsx',index_col=0)

df = pd.DataFrame({"A": data2['Rm-Rf'], "B": data2['SMB'], "C": data2['HML'], "D": data2['Ri']})
result = sm.ols(formula="D ~ A + B + C", data=df).fit()
print(result.params)
print(result.summary())
```

    Intercept    0.001722
    A            0.007210
    B           -0.949127
    C           -1.158802
    dtype: float64
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                      D   R-squared:                       0.303
    Model:                            OLS   Adj. R-squared:                  0.302
    Method:                 Least Squares   F-statistic:                     607.7
    Date:                Mon, 19 Mar 2018   Prob (F-statistic):               0.00
    Time:                        01:24:46   Log-Likelihood:                 10471.
    No. Observations:                4202   AIC:                        -2.093e+04
    Df Residuals:                    4198   BIC:                        -2.091e+04
    Df Model:                           3                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept      0.0017      0.000      4.893      0.000       0.001       0.002
    A              0.0072      0.021      0.339      0.735      -0.035       0.049
    B             -0.9491      0.028    -33.572      0.000      -1.005      -0.894
    C             -1.1588      0.039    -29.369      0.000      -1.236      -1.081
    ==============================================================================
    Omnibus:                      447.217   Durbin-Watson:                   1.820
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             3808.392
    Skew:                          -0.080   Prob(JB):                         0.00
    Kurtosis:                       7.661   Cond. No.                         128.
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    


```python
Alpha = result.params['Intercept']
B_MKT = result.params['A']
B_SMB = result.params['B']
B_HML = result.params['C']

df = pd.DataFrame({"Alpha":[intercept,Alpha],"B_MKT":[slope,B_MKT],
                   "B_SMB":['null',B_SMB], "B_HML":['null',B_HML]})
print(df)

df = pd.DataFrame(df, columns = ['Alpha','B_MKT','B_SMB', 'B_HML'], index = ['CAPM','3-factor'])
print(df)
```

          Alpha   B_HML     B_MKT     B_SMB
    0  0.000402    null  1.181969      null
    1  0.001722 -1.1588  0.007210 -0.949127
              Alpha  B_MKT B_SMB B_HML
    CAPM        NaN    NaN   NaN   NaN
    3-factor    NaN    NaN   NaN   NaN
    
