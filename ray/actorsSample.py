import ray

@ray.remote
class MyActor:
    def __init__(self):
        pass

    def run_task(self):
        return "Task completed"  # replace with your function

ray.init()

# Create actors with different resource demands
actor1 = MyActor.options(resources={"node:ip_1": 0.01}).remote()
actor2 = MyActor.options(resources={"node:ip_2": 0.01}).remote()

# Run tasks on different nodes
result1 = ray.get(actor1.run_task.remote())
result2 = ray.get(actor2.run_task.remote())
