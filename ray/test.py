from collections import Counter
import socket
import time

import ray

@ray.remote
class Actor:
    def __init__(self):
        pass

    def get_cluster_resources(self):
        return '''This cluster consists of
            {} nodes in total
            {} CPU resources in total
        '''.format(len(ray.nodes()), ray.cluster_resources())

    def task_def(self):
        time.sleep(0.001)
        # Return IP address.
        return socket.gethostbyname(socket.gethostname())

    def run_task(self):
        print('''TASK
----------------''')
        object_ids = [self.task_def for _ in range(10000)]
        ip_addresses = ray.get(object_ids)

        result = None
        print('Tasks executed')
        for ip_address, num_tasks in Counter(ip_addresses).items():
            result = '    {} tasks on {}'.format(num_tasks, ip_address)
        return result

ray.init()

# Create actors with different resource demands
actor1 = Actor.options(num_cpus=2).remote()
actor2 = Actor.options(num_gpus=1).remote()

print(ray.get(actor1.get_cluster_resources.remote()))
print(ray.get(actor1.run_task.remote()))
print(ray.get(actor2.get_cluster_resources.remote()))
print(ray.get(actor2.run_task.remote()))
