import os


def collect(cmd, prefix):
    r = os.popen(cmd)
    text = r.read()
    buffer = [prefix]
    for line in text.split("\n"):
        line = line.strip()
        if line == "":
            continue
        buffer.append(line)
    return " & ".join(buffer) + " \\\\"


for func in ("eps", "pbs", "be"):
    print(func)
    if func == "eps":
        sizes = [20, 40, 60, 80]
    elif func == "be":
        sizes = list(range(8, 13))
    else:
        sizes = [1]
    for size in sizes:
        if func != "pbs":
            print(f"m = {size}")
        try:
            for solver in ('hd', 'ct', 'tdsm'):
                cmd = f'python experiments.py -f {func} -s {solver} -m {size}'
                prefix = f"{solver.upper()}"
                buffer = collect(cmd, prefix)
                print(buffer)
        except:
            break

        print()
        print()
