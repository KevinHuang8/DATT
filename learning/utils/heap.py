import heapq

class heap(dict):
    """Dictionary that can be used as a priority queue.
    Keys of the dictionary are items to be put into the queue, and values
    are their respective priorities. All dictionary methods work as expected.
    The advantage over a standard heapq-based priority queue is
    that priorities of items can be efficiently updated (amortized O(1))
    using code as 'thedict[item] = new_priority.'
    The 'smallest' method can be used to return the object with lowest
    priority, and 'pop_smallest' also removes it.
    The 'sorted_iter' method provides a destructive sorted iterator.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._heapify()

    def _heapify(self):
        self._heap = [(v, k) for k, v in self.items()]
        heapq.heapify(self._heap)

    def find_min(self, return_value=False):
        """Return the item with the lowest priority.
        Raises IndexError if the object is empty.
        """
        
        heap = self._heap
        v, k = heap[0]
        while k not in self or self[k] != v:
            heapq.heappop(heap)
            v, k = heap[0]
        if return_value:
            return k, v
        return k

    def extract_min(self):
        """Return the item with the lowest priority and remove it.
        Raises IndexError if the object is empty.
        """
        
        heap = self._heap
        v, k = heapq.heappop(heap)
        while k not in self or self[k] != v:
            v, k = heapq.heappop(heap)
        del self[k]
        return k

    def __setitem__(self, key, val):
        # We are not going to remove the previous value from the heap,
        # since this would have a cost O(n).
        
        super().__setitem__(key, val)
        
        if len(self._heap) < 3 * len(self):
            heapq.heappush(self._heap, (val, key))
        else:
            # When the heap grows larger than 3 * len(self), we rebuild it
            # from scratch to avoid wasting too much memory.
            self._heapify()

    def setdefault(self, key, val):
        if key not in self:
            self[key] = val
            return val
        return self[key]

    def update(self, *args, **kwargs):
        # Reimplementing dict.update is tricky -- see e.g.
        # http://mail.python.org/pipermail/python-ideas/2007-May/000744.html
        # We just rebuild the heap from scratch after passing to super.
        
        super().update(*args, **kwargs)
        self._heapify()

    def sorted_iter(self):
        """Sorted iterator of the priority dictionary items.
        Beware: this will destroy elements as they are returned.
        """
        
        while self:
            yield self.extract_min()
