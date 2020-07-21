import time


def print_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        ret = func(*args, **kwargs)
        last = time.time() - start
        print("Lasts %.2fs" % last)
        return ret
    return wrapper

#@print_time
def hello(a, b):
    return a + b


if __name__ == '__main__':
    c = hello(3, 4)
    print(c)

    d = print_time(hello)(3, 4)