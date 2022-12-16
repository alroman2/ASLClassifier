
class Metric:
    """
    Base class for all metrics. A metric can be used to evaluate the performance of a model. 

     Errors:
        false positive: detecting an object when there is no object (e.g ground_truth bouding box is 0, but predicted is not 0)
        false negative: not detecting an object when there is an object (e.g ground_truth bouding box is not 0, but predicted is 0)

    Correct:
        true positive: detecting an object when there is an object (e.g ground_truth bouding box is not 0, and predicted is not 0)  
        true negative: not detecting an object when there is no object (e.g ground_truth bouding box is 0, and predicted is 0)

    Precision = True Positives / (True Positives + False Positives)
    Recall = True Positives / (True Positives + False Negatives)
    """

    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    def update(self, ground_truth, predicted):
        """
        Update the metric with the ground truth and the predicted values.

        """
        # false positive
        if len(ground_truth) == 0 and len(predicted) > 0:
            self.fp += 1
        # false negative
        elif len(ground_truth) > 0 and predicted is None:
            self.fn += 1
        # true positive
        elif len(ground_truth) > 0 and len(predicted) > 0:
            self.tp += 1
        # true negative
        elif len(ground_truth) == 0 and predicted is None:
            self.tn += 1

    def precision(self):
        """
        Calculate the precision.
        """
        if self.tp + self.fp == 0:
            return 0
        else:
            return self.tp / (self.tp + self.fp)

    def recall(self):
        """
        Calculate the recall.
        """
        if self.tp + self.fn == 0:
            return 0
        else:
            return self.tp / (self.tp + self.fn)

    def reset(self):
        """
        Reset the metric to its initial state.
        """
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
