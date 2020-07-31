# Interval Scoring
This repository contains python implementations of scoring rules for forecasts provided in a prediction interval format, for example when only quantiles of the predictive distribution are given.

- **Interval Score** (Gneiting & Raftery, 2007): Scores sharpness and calibration of a specific prediction interval. 
- **Weighted Interval Score** (Bracher, Gneiting & Reich, 2020): A weighted sum of sharpness and calibration scores of several prediction intervals. By suitable weighting, this can be used to approximate the CRPS.
- **Outside Interval Count**: Simple count of observations outside the prediction interval.
- **Interval Consistency Score**: Adapted version of the interval score which compares consecutive forecasts for the same point in time and evaluates their consistency (see below).

All credits for the interval score and weighted interval score go to the corresponding authors.

For exemplary uses of the scoring functions, see [here](#example). For the docstring of the functions, see [here](#docs).


#### References
- Gneiting, T. and A. E. Raftery (2007). Strictly proper scoring rules, prediction, and estimation. Journal of the American Statistical Association 102(477), 359–378, DOI: [10.1198/016214506000001437](https://doi.org/10.1198/016214506000001437)
- Bracher, J., Ray, E. L., Gneiting, T., & Reich, N. G. (2020). Evaluating epidemic forecasts in an interval format. arXiv preprint [arXiv:2005.12881](https://arxiv.org/abs/2005.12881).

## <a id='example'></a>Exemplary Usage of Scoring Functions
The following notebook snippets show the usage of different scoring functions with exemplary data.


```python
import timeit # timeit for measuring runtime of approaches
```


```python
import scoring as scoring # the scoring module
import numpy as np
```

### Test data


```python
# Exemplary ground truth observations as array
observations_test = np.array([4,7,4,6,2,1,3,8])

# Exemplary point forecasts
point_dict_test = {
    0.1: np.array([2, 4.7, 5.2 , 9.6, 1.8, -2  , 0.4, 8.8]),
    0.2: np.array([2, 4.7, 5.2 , 9.6, 1.8, -2  , 0.4, 8.8]),
    0.5: np.array([2, 4.7, 5.2 , 9.6, 1.8, -2  , 0.4, 8.8]),
    0.8: np.array([2, 4.7, 5.2 , 9.6, 1.8, -2  , 0.4, 8.8]),
    0.9: np.array([2, 4.7, 5.2 , 9.6, 1.8, -2  , 0.4, 8.8]),
}

# Exemplary quantile forecasts
quantile_dict_test = {
    0.1: np.array([2, 3  , 5   , 9  , 1  , -3  , 0.2, 8.7]),
    0.2: np.array([2, 4.6, 5   , 9.4, 1.4, -2  , 0.4, 8.8]),
    0.5: np.array([2, 4.7, 5.2 , 9.6, 1.8, -2  , 0.4, 8.8]),
    0.8: np.array([4, 4.8, 5.7 , 12 , 4.3, -1.5, 2  , 8.9]),
    0.9: np.array([5, 5  , 7   , 13 , 5  , -1  , 3  , 9])
}

# Exemplary alpha level
alpha_test=0.2
```

### Interval Score

Compute interval scores by providing a left and right quantile.


```python
scoring.interval_score(observations_test,alpha_test,q_left=quantile_dict_test[0.1],q_right=quantile_dict_test[0.9])
```




    (array([ 3. , 22. , 12. , 34. ,  4. , 22. ,  2.8,  7.3]),
     array([3. , 2. , 2. , 4. , 4. , 2. , 2.8, 0.3]),
     array([ 0., 20., 10., 30.,  0., 20.,  0.,  7.]))



Compute interval scores by providing a dictionary of quantiles. The quantiles needed for the alpha level specified are automatically selected from the dictionary.


```python
scoring.interval_score(observations_test,alpha_test,q_dict=quantile_dict_test)
```




    (array([ 3. , 22. , 12. , 34. ,  4. , 22. ,  2.8,  7.3]),
     array([3. , 2. , 2. , 4. , 4. , 2. , 2.8, 0.3]),
     array([ 0., 20., 10., 30.,  0., 20.,  0.,  7.]))



Test if the percentage version of the interval score works:
We use the (1-[alpha=1])-interval, i.e. the median, for the interval score, set `percentage=True` and check whether this is two times the absolute percentage error of the median.


```python
print(scoring.interval_score(observations_test,1,q_dict=quantile_dict_test,percent=True)[0])
print(2*np.abs((observations_test-quantile_dict_test[0.5])/observations_test))
```

    [1.         0.65714286 0.6        1.2        0.2        6.
     1.73333333 0.2       ]
    [1.         0.65714286 0.6        1.2        0.2        6.
     1.73333333 0.2       ]
    

### Weighted Interval Score

Compute weighted interval scores by providing a dictionary of quantiles and a list of alpha levels and corresponding weights.

There are two implementations, a simple one based on iterated computation of interval scores and a faster one based on joint computation of all interval scores. They yield the same results, so the faster one is preferrable and the basic one is only provided for understanding purposes.


```python
print(scoring.weighted_interval_score(observations_test,alphas=[0.2,0.4],weights=[2,5],q_dict=quantile_dict_test))
print(scoring.weighted_interval_score_fast(observations_test,alphas=[0.2,0.4],weights=[2,5],q_dict=quantile_dict_test))
```

    (array([ 2.28571429, 14.28571429,  7.5       , 23.71428571,  3.21428571,
           15.57142857,  5.51428571,  5.01428571]), array([2.28571429, 0.71428571, 1.07142857, 3.        , 3.21428571,
           0.92857143, 1.94285714, 0.15714286]), array([ 0.        , 13.57142857,  6.42857143, 20.71428571,  0.        ,
           14.64285714,  3.57142857,  4.85714286]))
    (array([ 2.28571429, 14.28571429,  7.5       , 23.71428571,  3.21428571,
           15.57142857,  5.51428571,  5.01428571]), array([2.28571429, 0.71428571, 1.07142857, 3.        , 3.21428571,
           0.92857143, 1.94285714, 0.15714286]), array([ 0.        , 13.57142857,  6.42857143, 20.71428571,  0.        ,
           14.64285714,  3.57142857,  4.85714286]))
    

Compute weighted interval scores by providing a dictionary of quantiles and a list of alpha levels, but no corresponding weights. This means setting `weights=alphas/2`, yielding an approximation of the CRPS.


```python
scoring.weighted_interval_score_fast(observations_test,alphas=[0.2,0.4],weights=None,q_dict=quantile_dict_test)
```




    (array([ 2.33333333, 14.8       ,  7.8       , 24.4       ,  3.26666667,
            16.        ,  5.33333333,  5.16666667]),
     array([2.33333333, 0.8       , 1.13333333, 3.06666667, 3.26666667,
            1.        , 2.        , 0.16666667]),
     array([ 0.        , 14.        ,  6.66666667, 21.33333333,  0.        ,
            15.        ,  3.33333333,  5.        ]))



Compute percentage version of the weighted interval scores (once with just the median, equal to the simple interval scorce, and once with several intervals).


```python
print(scoring.weighted_interval_score_fast(observations_test,alphas=[1],weights=None,q_dict=point_dict_test,percent=True)[0])
print(scoring.weighted_interval_score_fast(observations_test,alphas=[0.2,0.4,1],weights=None,q_dict=point_dict_test,percent=True)[0])
```

    [1.         0.65714286 0.6        1.2        0.2        6.
     1.73333333 0.2       ]
    [ 1.875       1.23214286  1.125       2.25        0.375      11.25
      3.25        0.375     ]
    

#### Compare runtimes of weighted interval score methods

Runtimes of simple implementation for different sizes


```python
print({n: min(timeit.repeat(lambda: scoring.weighted_interval_score(observations_test,alphas=[0.2]*n,weights=None,q_dict=quantile_dict_test),repeat=100,number=100)) for n in [2,5,10,20]})
```

    {2: 0.007953100000000823, 5: 0.014574499999999269, 10: 0.02527900000000116, 20: 0.04693550000000002}
    

Runtimes of fast implementation for different sizes


```python
print({n: min(timeit.repeat(lambda: scoring.weighted_interval_score_fast(observations_test,alphas=[0.2]*2,weights=None,q_dict=quantile_dict_test),repeat=100,number=100)) for n in [2,5,10,20,40,80]})
```

    {2: 0.008790000000001186, 5: 0.00875909999999891, 10: 0.008776400000002127, 20: 0.008793000000000717, 40: 0.008763800000000543, 80: 0.008803100000001507}
    

As can be seen, the fast implementation is in fact quicker for more than 2 or 3 intervals. Its runtime stays almost constant.

### Outside Interval Count

Count number of times the true observation was outside an interval.


```python
print(scoring.outside_interval(observations_test,lower=quantile_dict_test[0.1],upper=quantile_dict_test[0.9]))
```

    [0 1 1 1 0 1 0 1]
    

### Interval Consistency Score

Compute adapted variant of the interval score which measures the consistency of updated intervals over time.

The interval consistency score does not evaluate the sharpness of the intervals. It only penalizes the difference between old and new forecasted intervals when the new interval is (partially) outside the old interval.
Ideally, updated predicted intervals would always be within the previous estimates of the interval, yielding
a score of zero (best). The underlying logic is that the old forecasts have been made with less perfect knowledge than the new ones and should reflect this epistemic uncertainty in wider prediction intervals.


```python
print(scoring.interval_consistency_score(lower_old=quantile_dict_test[0.1],upper_old=quantile_dict_test[0.9],lower_new=quantile_dict_test[0.1],upper_new=quantile_dict_test[0.8]))
```

    [0. 0. 0. 0. 0. 0. 0. 0.]
    


```python
print(scoring.interval_consistency_score(lower_old=quantile_dict_test[0.1],upper_old=quantile_dict_test[0.8],lower_new=quantile_dict_test[0.1],upper_new=quantile_dict_test[0.9]))
```

    [1.  0.2 1.3 1.  0.7 0.5 1.  0.1]
    

## <a id='docs'></a>Function Docs

### interval_score
    Compute interval scores for an array of observations and predicted intervals.
    
    Either a dictionary with the respective (alpha/2) and (1-(alpha/2)) quantiles via q_dict needs to be specified or the quantiles need to be specified via q_left and q_right.

    Parameters
    ----------
    observations : array_like
        Ground truth observations.
    alpha : numeric
        Alpha level for (1-alpha) interval.
    q_dict : dict, optional
        Dictionary with predicted quantiles for all instances in `observations`.
    q_left : array_like, optional
        Predicted (alpha/2)-quantiles for all instances in `observations`.
    q_right : array_like, optional
        Predicted (1-(alpha/2))-quantiles for all instances in `observations`.
    percent: bool, optional
        If `True`, score is scaled by absolute value of observations to yield a percentage error. Default is `False`.
    check_consistency: bool, optional
        If `True`, quantiles in `q_dict` are checked for consistency. Default is `True`.

    Returns
    -------
    total : array_like
        Total interval scores.
    sharpness : array_like
        Sharpness component of interval scores.
    calibration : array_like
        Calibration component of interval scores.
        
### weighted_interval_score / weighted_interval_score_fast
    Compute weighted interval scores for an array of observations and a number of different predicted intervals.
    
    This function implements the WIS-score. A dictionary with the respective (alpha/2)
    and (1-(alpha/2)) quantiles for all alpha levels given in `alphas` needs to be specified.
    
    Parameters
    ----------
    observations : array_like
        Ground truth observations.
    alphas : iterable
        Alpha levels for (1-alpha) intervals.
    q_dict : dict
        Dictionary with predicted quantiles for all instances in `observations`.
    weights : iterable, optional
        Corresponding weights for each interval. If `None`, `weights` is set to `alphas`, yielding the WIS^alpha-score.
    percent: bool, optional
        If `True`, score is scaled by absolute value of observations to yield the double absolute percentage error. Default is `False`.
    check_consistency: bool, optional
        If `True`, quantiles in `q_dict` are checked for consistency. Default is `True`.
        
    Returns
    -------
    total : array_like
        Total weighted interval scores.
    sharpness : array_like
        Sharpness component of weighted interval scores.
    calibration : array_like
        Calibration component of weighted interval scores.
        
### interval_consistency_score
    Compute interval consistency scores for an old and a new interval.
    
    Adapted variant of the interval score which measures the consistency of updated intervals over time.
    Ideally, updated predicted intervals would always be within the previous estimates of the interval, yielding
    a score of zero (best).
    
    Parameters
    ----------
    lower_old : array_like
        Previous lower interval boundary for all instances in `observations`.
    upper_old : array_like, optional
        Previous upper interval boundary for all instances in `observations`.
    lower_new : array_like
        New lower interval boundary for all instances in `observations`. Ideally higher than the previous boundary.
    upper_new : array_like, optional
        New upper interval boundary for all instances in `observations`. Ideally lower than the previous boundary.
    check_consistency: bool, optional
        If interval boundaries are checked for consistency. Default is `True`.
        
    Returns
    -------
    scores : array_like
        Interval consistency scores.
        
### outside_interval
    Indicate whether observations are outside a predicted interval for an array of observations and predicted intervals.
    
    Parameters
    ----------
    observations : array_like
        Ground truth observations.
    lower : array_like, optional
        Predicted lower interval boundary for all instances in `observations`.
    upper : array_like, optional
        Predicted upper interval boundary for all instances in `observations`.
    check_consistency: bool, optional
        If `True`, interval boundaries are checked for consistency. Default is `True`.
        
    Returns
    -------
    Out : array_like
        Array of zeroes (False) and ones (True) counting the number of times observations where outside the interval.


```python

```
