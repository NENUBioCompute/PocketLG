import threading
import time
import os
from queue import Queue


""" 
thread_list = {
     [target,parameter],
}
"""


class thread_manager(object):
    """ class decorator"""
    def __init__(self, func):
        self.func = func

    def __call__(self, address):
        thread_list = self.func(address)
        """decorated function returns thread_list  """
        threads = []
        results = []
        q = Queue()
        num = 0
        """use queue to save results"""
        for threadName in thread_list:
            num += 1
            thread = threading.Thread(target=threadName[0], args=(threadName[1], q))
            thread.start()
            # thread.join()
            threads.append(thread)
        for thread in threads:
            thread.join()
        while q.empty()==False:
            results.append(q.get())
        return results
