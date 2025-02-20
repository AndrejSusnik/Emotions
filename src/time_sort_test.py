import time
import random

arr = list(range(1000))
start_time = time.time()
arr.sort()
end_time = time.time()

# print("Sorted:", arr)
print(f"Time taken: {end_time - start_time:.10f} seconds")


arr = list(range(1000))
random.shuffle(arr)
start_time = time.time()
arr.sort()
end_time = time.time()

# print("Sorted:", arr)
print(f"Time taken: {end_time - start_time:.10f} seconds")