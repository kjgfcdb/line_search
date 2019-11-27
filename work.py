import os


for func in ('tri', 'eps', 'er', 'pi'):
    for solver in ('lbfgs', 'clbfgs'):
        for m in [5, 9, 15]:
            cmd = f'python experiments.py -f {func} -s {solver} -n 1000 -m {m}'
            os.system(cmd)
    for c in [1, 2]:
        cmd = f'python experiments.py -f {func} -s "in" -n 1000 -c {c}'
        os.system(cmd)
