def interp12(cur_step_size, g, G):
	# 一点二次插值法（牛顿法）
	# cur_step_size: 当前alpha
	# g/G: alpha的一阶导数/二阶导数
	# NOTE: 这里的导数指的是函数phi(alpha) = f(x + alpha*d)关于参数alpha的导数
	nxt_step_size = cur_step_size - g/G
	return nxt_step_size