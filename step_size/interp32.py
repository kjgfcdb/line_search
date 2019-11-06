def interp32(stepsize1, stepsize2, stepsize3, phi):
	# 三点二次插值法
	# stepsize1: alpha1，通常情况下是alpha_{k}
	# stepsize2: alpha2，通常情况是alpha_{k-1}
	# stepsize3: alpha3，通常情况是alpha_{k-2}
	# phi: 函数phi(alpha) = f(x + alpha*d)，要求返回结果为(函数值，导数)形式
	v1, _ = phi(stepsize1)
	v2, _ = phi(stepsize2)
	v3, _ = phi(stepsize3)
	nxt_stepsize = 0.5*(stepsize1 + stepsize2) + 0.5*(v1-v2)*(stepsize2 - stepsize3)*(stepsize3 - stepsize1) / \
            ((stepsize2 - stepsize3)*v1 + (stepsize3 - stepsize1)
             * v2 + (stepsize1 - stepsize2)*v3)
