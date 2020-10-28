Simulation of an ensemble of SPM models using multiple Python processes using shared memory or message passing.

## Files
Module `sharedarray.py` implements the first shared memory approach, making use of the SharedArray package, available on PyPI. A pool of processes is spawned at each step, i.e:
```python
for time in range(0, end_time, dt)
    ...
    with ProcessPoolExecutor() as ex:
    ex.map(shm_step, work)
    ...
```

Module `pool.py` implements a message passing approach, using `multiprocessing.map`.
In this case processes work through the total number of timesteps `Nsteps` before 
returning:
```python
with Pool(processes) as p:
        solutions = p.map(worker, work)
```
and 
```python
def worker(work):
    solution = work.sol_init
    for t in range(0, work.end_time, work.dt):
        solution = work.do_step(solution)

    return solution
```

Module `solve.py` is also based on `multiprocessing.Pool.map`, but the worker function 
solves the SPMs using `BaseSolver.solve()` rather than stepping using `BaseSolver.step()`
in a loop.

Module `serial.py` implements a serial version: no `map`, `multiprocessing` or `futures` there, just good old single threaded python.

## Tests
All implementations are validated against a reference implementation (`ref/ref.py`).
See the tests in `tests.py`.

Edit 28/10/2020: Currently `solve_w_Pool_solve` is not tested.

## Benchmarking

The script `bench.py` can be used to time (wall time) the execution of the various implementations over a range of processes number. For each of the processes number, the execution is repeated `Nreps` times. Specific implementation(s) can be specified as command line options. No options will run each implementation in a sequence.

```shell
$ python bench.py --help
usage: bench.py [-h] [--sharedarray] [--pool] [--serial] [--solve]

Time the resolution of an ensemble of SPMe models, for various number of processes and report timings. Each implementation is executed 10 times. This is done
following different implementations, see the list of options below. Nspm = 8, Nsteps = 10, dt = 1, nproc_range = [2, 4]

optional arguments:
  -h, --help     show this help message and exit
  --sharedarray  Use implementation based on SharedArray package.
  --pool         Use implementation based on multiprocessing.Pool.map
  --serial       Serial implementation, i.e. solve SPMs in a sequence.
  --solve        Same as --pool but worker function calls BaseSolver.solve() instead of stepping the model with BaseSolver.step().
```
