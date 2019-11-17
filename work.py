import os


for func in ('powell_badly_scaled', 'extended_powell_singular', 'biggs_exp6'):
    for solver in ('damp_newton', 'stable_newton', 'fletcher_freeman'):
        for ls in ('armijo_goldstein', 'wolfe_powell', 'gll'):
            cmd = f'python experiments.py -s {solver} -l {ls} -f {func} -c'
            os.system(cmd)
