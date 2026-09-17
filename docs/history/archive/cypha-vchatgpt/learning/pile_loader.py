from datasets import load_dataset

class PileLoader:
    def __init__(self, split="train", streaming=True, cache_dir=None):
        self.ds = load_dataset("CarperAI/pilev2-dev", split=split, streaming=streaming, cache_dir=cache_dir)
    def stream(self):
        for row in self.ds:
            yield row["text"]
    def get_batch(self, batch_size=16):
        stream = self.stream()
        batch = []
        for t in stream:
            batch.append(t)
            if len(batch) == batch_size:
                yield batch
                batch = []
