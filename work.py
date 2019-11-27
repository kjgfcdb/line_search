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
    for solver in ('lbfgs', 'clbfgs'):
        for m in [5, 9, 15]:
            cmd = f'python experiments.py -f {func} -s {solver} -n 1000 -m {m}'
            prefix = solver + "_" + str(m)
            buffer = collect(cmd, prefix)
            print(buffer)
    for c in [1, 2]:
        cmd = f'python experiments.py -f {func} -s "in" -n 1000 -c {c}'
        prefix = solver + "_" + str(c)
        buffer = collect(cmd, prefix)
        print(buffer)
    
    print()
    print()
