"""
收集实验结果并输出到csv文件
"""
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
    return ", ".join(buffer)

os.makedirs("csv", exist_ok=True)

for func in ("eps", "pbs", "be6"):
    print(func)
    if func == "eps":
        sizes = [20, 40, 60, 80]
    elif func == "be6":
        sizes = list(range(8, 13))
    else:
        sizes = [1]
    for size in sizes:
        g = open(f"csv/{func}_{size}.csv", "w")
        if func == "pbs":
            g.write(f"PBS,最优解,函数值,梯度范数,迭代数,函数调用次数\n")
        elif func == "eps":
            g.write(f"{func.upper()},最优解范数,最优解均值,函数值,梯度范数,迭代数,函数调用次数\n")
        elif func == "be6":
            g.write(f"{func.upper()},最优解,函数值,梯度范数,迭代数,函数调用次数\n")
        for solver in ('hd', 'ct', 'tdsm'):
            cmd = f'python experiments.py -f {func} -s {solver} -m {size}'
            prefix = f"{solver.upper()}"
            buffer = collect(cmd, prefix)
            g.write(buffer + "\n")
        for solver in ("ff", "sn", "dn"):
            for ls_func in ["armijo_goldstein", "wolfe_powell"]:
                cmd = f'python experiments.py -f {func} -s {solver} -m {size} -ls {ls_func}'
                prefix = f"{solver.upper()}-{ls_func[0].upper()}"
                buffer = collect(cmd, prefix)
                g.write(buffer + "\n")
        g.close()
