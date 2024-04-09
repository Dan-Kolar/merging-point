# merging-point
Implementation of the merging point algorithm from the paper [Wealth Survey Calibration Using Income Tax Data](https://ideas.repec.org/p/inq/inqwps/ecineq2023-659.html).

## Requirements

```python
pip install -r requirements.txt
```

The code was tested in Python versions 3.8.19 and 3.12.2. In case of issues, version-specific *requirements* files are provided.

## Run test

```python
python mp_script.py
```

*mp_script.py* (as well as *mp_script_notebook.ipynb*) defines the *mp* function which outputs candidate merging points and a grid dataset to create graphs. The script includes a test simulation using simulated sample data.

## Inputs

* survey microdata with a weight column and an income column
* tax data in the form of an output file from the [Generalized Pareto interpolation](https://wid.world/gpinter/) (gpinter) programme; the tax distribution should be continuous
* the total (adult) population size that is consistent with both the survey and the tax data

Please adjust lines 52-54 in *mp_script.py* to upload the two datasets and specify the total population, and then run

```python
python mp_script.py
```


## Arguments

Arguments can be specified on line 55 in *mp_script.py*

|  Arguments   | Details  | Default value | 
|  ----  | ----  | ----  |
| ts  | percentile from which tax data are considered reliable (trustable span); min:0, max:0.98 | 0 |
| inccol  | income column in the survey | "income" |
| weightcol | weight column in the survey | "weight" |
| gamma | distance parameter gamma (see paper) | 3 |
| beta | distance parameter beta (see paper) | 30 |
| other_mp | list of additional percentiles to be included among candidate merging points | empty list |
