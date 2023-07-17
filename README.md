### MO_Ring_PSO_SCD: Multi-objective particle swarm optimizer using ring topology and special crowding distance

##### Reference: Yue C, Qu B, Liang J. A multiobjective particle swarm optimizer using ring topology for solving multimodal multiobjective problems[J]. IEEE Transactions on Evolutionary Computation, 2017, 22(5): 805-817.

##### The MO_Ring_PSO_SCD belongs to the category of multi-objective evolutionary algorithms (MOEAs). MO_Ring_PSO_SCD is a powerful algorithm to solve the multi-modal multi-objective optimization (MMO) problems.

| Variables | Meaning                                  |
| --------- | ---------------------------------------- |
| npop      | Population size                          |
| iter      | Iteration number                         |
| lb        | Lower bound                              |
| ub        | Upper bound                              |
| omega     | Inertia weight (default = 0.7298)        |
| c1        | Acceleration constant 1 (default = 2.05) |
| c2        | Acceleration constant 2 (default = 2.05) |
| n_NBA     | Maximum NBA size (default = 15)          |
| n_PBA     | Maximum PBA size (default = 5)           |
| dim       | Dimension                                |
| pos       | Position                                 |
| vmin      | Minimum velocity                         |
| vmax      | Maximum velocity                         |
| vel       | Velocity                                 |
| objs      | Objectives                               |
| nobj      | Objective number                         |
| PBA       | Personal best archive                    |
| NBA       | Neighborhood best archive                |
| PBA_objs  | The objectives of PBA                    |
| NBA_objs  | The objective of NBA                     |
| scd       | Special crowding distance                |
| pf        | Pareto front                             |
| ps        | Pareto set                               |

#### Test problem: MMF1



$$
\left\{
\begin{aligned}
&f_1(x)=|x_1-2|\\
&f_2(x)=1-\sqrt{|x_1 - 2|}+2(x_2-\sin{(6 \pi |x_1 - 2| + \pi)})^2\\
&1 \leq x_1 \leq 3, -1 \leq x_2 \leq 1
\end{aligned}
\right.
$$



#### Example

```python
if __name__ == '__main__':
    t_npop = 800
    t_iter = 100
    t_lb = np.array([1, -1])
    t_ub = np.array([3, 1])
    main(t_npop, t_iter, t_lb, t_ub)
```

##### Output:

![](https://github.com/Xavier-MaYiMing/MO_Ring_PSO_SCD/blob/main/Pareto%20front.png)

![](https://github.com/Xavier-MaYiMing/MO_Ring_PSO_SCD/blob/main/Pareto%20front.png)

```python
Iteration 10 completed.
Iteration 20 completed.
Iteration 30 completed.
Iteration 40 completed.
Iteration 50 completed.
Iteration 60 completed.
Iteration 70 completed.
Iteration 80 completed.
Iteration 90 completed.
Iteration 100 completed.
```

