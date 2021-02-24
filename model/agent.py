import numpy as np
import onnxruntime


class Agent(object):

    """
    Class of agent for play minesweeper
    Input parameters:
      - dicision_field - Size for decision field side
      - checkpoint_dir - Path where model is saved
    """

    def __init__(self, decision_field, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir  # Path where model is saved
        self.decision_field = decision_field  # Size for decision field side
        self.model = onnxruntime.InferenceSession(checkpoint_dir)  # Neural net model

    # Select action
    def get_action(self, history):
        # Get prediction from model [1 x decision_field x decision_field x 1]
        history = np.expand_dims(history, axis=0)
        history = history if isinstance(history, list) else [history]
        feed = dict([(inputs.name, history[n]) for n, inputs in enumerate(self.model.get_inputs())])
        q_value = self.model.run(None, feed)

        return q_value
