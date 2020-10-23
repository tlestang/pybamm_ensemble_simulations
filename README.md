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

Module `serial.py` implements a serial version: no `map`, `multiprocessing` or `futures` there, just good old single threaded python.

## Tests
All implementations are validated against a reference implementation (`ref/ref.py`).
See the tests in `tests.py`.

## Benchmarking

The script `bench.py` can be used to time (wall time) the execution of the various implementations over a range of processes number. For each of the processes number, the execution is repeated `Nrep` times.
