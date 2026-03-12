import numpy as np
import pandas as pd
import multiprocessing as mp
import math
import os
from multiprocessing import shared_memory
from src.gof_methods import ks_test, ad_test, chisq_test, g_test

def worker_compare(base_cols, shm_base, new_cols, shm_new, dtype_base, dtype_new, 
                   base_shape, new_shape, start, stop, dist1, dist2, 
                   delimiter=127, hist_bin=10):

    base_arr = np.ndarray(base_shape, dtype=dtype_base, buffer=shm_base.buf)
    new_arr = np.ndarray(new_shape, dtype=dtype_new, buffer=shm_new.buf)

    results = []
    #logs = []

    for i in range(start, stop):
        for j in range(len(new_cols)):
            col1 = base_cols[i]
            col2 = new_cols[j]

            nuniq1 = len(set(base_arr[:, i]))
            nuniq2 = len(set(new_arr[:, j]))

            if (nuniq1 < 2):
                break;
            elif (nuniq2 < 2):
                continue

            # Check the threshold 
            if (nuniq1 <= delimiter and nuniq2 <= delimiter):
                uniq1 = set(base_arr[:, i])
                uniq2 = set(new_arr[:, j])

                if len(uniq1.intersection(uniq2)) < 2:
                    continue
                res = chisq_test(base_arr[:, i], new_arr[:, j])
                results.append([dist1, dist2, col1, col2, 'CHISQ', res.statistic, res.pvalue])
                res = g_test(base_arr[:, i], new_arr[:, j])
                results.append([dist1, dist2, col1, col2, 'G', res.statistic, res.pvalue])

            elif (nuniq1 > delimiter and nuniq2 > delimiter):
                res = ks_test(base_arr[:, i], new_arr[:, j], hist_bin)
                results.append([dist1, dist2, col1, col2, 'KS', res.statistic, res.pvalue])
                res = ad_test(base_arr[:, i], new_arr[:, j], hist_bin)
                results.append([dist1, dist2, col1, col2, 'AD', res.statistic, res.pvalue])

    # write per-worker files - avoid conflict
    pid = os.getpid()
    result_columns = ['val1', 'val2', 'attr1','attr2','test','statistic','p-value']
    pd.DataFrame(results, columns=pd.Index(result_columns))\
      .to_csv(f"./comparisons/comparisons_{pid}.csv", index=False, header=False)


def parallel_cart_prod(base_cols, base_data, new_cols, new_data,
                       dist1, dist2, delimiter=127, hist_bin=10, num_workers=None):
    if num_workers is None:
        num_workers = mp.cpu_count() - 4

    os.system("mkdir -p comparisons")
    print(f"== Delimiter: {delimiter},  Bin: {hist_bin}")

    # Start and copy input shared memory
    shm_base = shared_memory.SharedMemory(create=True, size=base_data.nbytes)
    shm_new = shared_memory.SharedMemory(create=True, size=new_data.nbytes)
    base_sh = np.ndarray(base_data.shape, dtype=base_data.dtype, buffer=shm_base.buf)
    new_sh = np.ndarray(new_data.shape, dtype=new_data.dtype, buffer=shm_new.buf)
    base_sh[:] = base_data[:]
    new_sh[:] = new_data[:]

    # Divide base_data in chunks for each process
    n = len(base_cols)
    chunk = math.ceil(n / num_workers)

    processes = []
    for id in range(num_workers):
        start = id * chunk
        if start >= n:
            break
        stop = min((id+1) * chunk, n)
        p = mp.Process(target=worker_compare,
                       args=(base_cols, shm_base, new_cols, shm_new, base_data.dtype, 
                             new_data.dtype, base_data.shape, new_data.shape, 
                             start, stop, dist1, dist2, delimiter, hist_bin))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Cleanup shared memory
    shm_base.close()
    shm_base.unlink()
    shm_new.close()
    shm_new.unlink()

    # Merge results
    os.system("echo \"val1,val2,attr1,attr2,test,statistic,p-value\" > comparisons.csv")
    os.system("cat ./comparisons/comparisons_* | sort >> comparisons.csv")
