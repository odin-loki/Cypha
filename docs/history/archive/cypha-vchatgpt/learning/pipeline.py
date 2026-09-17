from queue import Queue
import random

class DataPipeline:
    def __init__(self, batch_size=32):
        self.q = Queue()
        self.batch_size = batch_size
    def put(self, x, y):
        self.q.put((x, y))
    def batch(self):
        items = []
        for _ in range(self.batch_size):
            try:
                items.append(self.q.get_nowait())
            except:
                break
        if not items: return None, None
        X, Y = zip(*items)
        return X, Y
    def augment(self, x):
        if random.random() > 0.5: return x
        return [xi + random.gauss(0,0.1) for xi in x]