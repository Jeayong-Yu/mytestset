
# 경희대학교 금융공학 연구실 Jupyter notebook tutorial

경희대학교 금융공학 연구실 Jupyter notebook tutorial을 위해 작성한 markdown 입니다.

## 1. live code


```python
3**2+4**2
```




    25




```python
def square_sum(a, b):
    sol = a**2+b**2
    return sol
square_sum(3,4)
```




    25




```python
import numpy as np
```


```python
np.zeros(3)
```




    array([ 0.,  0.,  0.])



## 2. equation


```python
# square_sum 함수는 a**2+b**2를 구하는 함수입니다.
```

square_sum 함수는 \\(a^2+b^2\\)를 구하는 함수 입니다.

[서비스를 이용하시려면 여기를 클릭 하세요](https://www.wolframalpha.com/)

## 3. visualization
### 3.1 image


```python
from IPython.display import Image
Image('http://ncc.phinf.naver.net/20160921_117/1474417831859Oe6xy_PNG/01.png')

```




![png](output_12_0.png)



### 3.2 video


```python
from IPython.display import YouTubeVideo
YouTubeVideo('SLPo0o_HArs')

```





        <iframe
            width="400"
            height="300"
            src="https://www.youtube.com/embed/SLPo0o_HArs"
            frameborder="0"
            allowfullscreen
        ></iframe>
        



## 4. performance check


```python
%timeit 3**2+4**2
```

    10000000 loops, best of 3: 15.6 ns per loop
    


```python
%timeit square_sum(3,4)
```

    1000000 loops, best of 3: 796 ns per loop
    


```python

```
