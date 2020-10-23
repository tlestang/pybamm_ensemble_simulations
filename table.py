def make_table(benchmark_results, nproc_range):
    impl_col = []
    nproc_col = []
    avg_col = []
    err_col = []
    for impl in benchmark_results.keys():
        if impl == "serial":
            impl_col.append(impl)
            nproc_col.append("-")
            avg_col.append(np.mean(benchmark_results[impl]))
            err_col.append(
                np.std(benchmark_results[impl]) / np.sqrt(len(benchmark_results[impl]))
            )
        else:
            impl_col.extend([impl] * len(list(nproc_range)))
            nproc_col.extend([nproc for nproc in nproc_range])
            avg_col.extend(
                [
                    np.mean(individual_times)
                    for individual_times in benchmark_results[impl]
                ]
            )
            err_col.extend(
                [
                    np.std(individual_times) / np.sqrt(len(individual_times))
                    for individual_times in benchmark_results[impl]
                ]
            )
    table = PrettyTable()
    table.add_column("Implementation", impl_col)
    table.add_column("# processes", nproc_col)
    table.add_column("Average time (s)", avg_col)
    table.add_column("Standard error (s)", err_col)

    return table
