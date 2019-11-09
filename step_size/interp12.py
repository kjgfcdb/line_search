def interp12(cur_step_size, g, G):
	"""一点二次插值法（牛顿法）
	Parameters
	----------
	cur_step_size : float
		当前步长
	g : 当前函数的一阶导数
	G : 当前函数的二阶矩阵
	Returns
	-------
	返回下一个步长
	"""
	nxt_step_size = cur_step_size - g/G
	return nxt_step_size