import numpy as np
import sys

class StreamTrainer:
    def __init__(self, pipeline, metrics=None):
        self.pipeline = pipeline   # the pipeline object
        self.metrics = metrics   # StreamingMetrics object
        self.log = []       # list of dicts and one per chunk
        self.chunk_count = 0

    def fit_chunk(self, X, y):
        self.pipeline.partial_fit(X, y) # train the pipeline on this specific block of data
        
        self.chunk_count += 1 # then, increse chunk tracker by 1
        
        memory_footprint = sys.getsizeof(X) / 1024 / 1024 #for memory footprint
        
        # append record to the history log
        self.log.append({
            'chunk': self.chunk_count,
            'type': 'fit',
            'memory_footprint': memory_footprint
        })
        return self
    
    def score_chunk(self, X, y):
        # make predict through the pipeline
        y_pred = self.pipeline.predict(X)
        
        if self.metrics is not None:
            self.metrics.update(y, y_pred)
            results = self.metrics.result()
        else:
            results = {}

        y_arr = np.asarray(y)
        if y_arr.size > 0:
            chunk_accuracy = float(np.mean(y_arr == y_pred))
        else:
            chunk_accuracy = 0

        # update log record with scoring metrics
        memory_footprint = sys.getsizeof(X) / 1024 / 1024
        
        # capture cumulative accuracy from metrics object if available
        cumulative_accuracy = results.get('accuracy', chunk_accuracy)

        self.log.append({
            'chunk': self.chunk_count,
            'type': 'score',
            'chunk_accuracy': chunk_accuracy,
            'cumulative_accuracy': cumulative_accuracy,
            'memory_footprint': memory_footprint
        })

        return results
    
    def get_log(self):
        return self.log
    
    def reset(self):
        # reset log history, reset counts and external metrics if have
        self.log = []
        self.chunk_count = 0
        if self.metrics is not None:
            self.metrics.reset()