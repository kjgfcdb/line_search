import os


def collect(cmd, prefix):
    r = os.popen(cmd)
    text = r.read()
    buffer = []
    buffer.append(prefix)
    for line in text.split("\n"):
        line = line.strip()
        if line == "":
            continue
        buffer.append(line)
    return " & ".join(buffer) + " \\\\"


for func in ('tri', 'eps', 'er', 'pi'):
    print(func)
    if func == 'eps':
        sizes = [100, 1000, 5000, 10000]
    else:
        sizes = [1000, 5000, 10000]
    for size in sizes:
        print(size)
        for solver in ('lbfgs', 'clbfgs'):
            for m in [5, 9, 15]:
                cmd = f'python experiments.py -f {func} -s {solver} -n {size} -m {m}'
                prefix = solver.upper() + "\\_" + str(m)
                buffer = collect(cmd, prefix)
                print(buffer)
        for c in [1, 2]:
            cmd = f'python experiments.py -f {func} -s "in" -n {size} -c {c}'
            prefix = "IN\\_" + str(c)
            buffer = collect(cmd, prefix)
            print(buffer)

        print()
        print()
