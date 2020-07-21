from multiprocessing.pool import Pool
from jiangziya.utils.print_util import print_time

@print_time
def parallel_process(func, dataset):
	pool = Pool(4)
	result = [pool.apply_async(func, data) for data in dataset]
	pool.close()
	pool.join()
	return [_result.get() for _result in result]

@print_time
def serial_process(func, dataset):
	result = [func(data) for data in dataset]
	return result

def add(x):
	return x ** 2


if __name__ == '__main__':
	dataset = range(10000)

	result = serial_process(add, dataset)
	print('serial result', result[-10:])

	result = parallel_process(add, dataset)
	print('parallel result', result[-10:])

