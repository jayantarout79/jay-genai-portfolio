import time
from Custom_Functions.decorators_demo import log_execution, time_execution

@log_execution
def say_hello(name):
    return f"Hello, {name}!"

@time_execution
def slow_add(a, b):
    time.sleep(1.5)  # simulate delay
    return a + b

if __name__ == "__main__":
    print(say_hello("Jayanta"))
 #   print(slow_add(5, 7))


 #python3 -m Simple-Python.simple_function_decorator_test