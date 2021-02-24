import numpy as np
from model.agent import Agent


def create_fields(history, pad_horizontal, pad_vertical):
    # Create four corner fields for more precise prediction
    # Top left corner
    field_1 = np.concatenate((history, pad_horizontal), axis=1)
    field_1 = np.concatenate((field_1, pad_vertical), axis=0)
    # Top right corner
    field_2 = np.concatenate((pad_horizontal, history), axis=1)
    field_2 = np.concatenate((field_2, pad_vertical), axis=0)
    # Bottom left corner
    field_3 = np.concatenate((history, pad_horizontal), axis=1)
    field_3 = np.concatenate((pad_vertical, field_3), axis=0)
    # Bottom right corner
    field_4 = np.concatenate((pad_horizontal, history), axis=1)
    field_4 = np.concatenate((pad_vertical, field_4), axis=0)

    return [field_1, field_2, field_3, field_4]


def get_predict(agent, history):
    # Get coordinates of next step
    # Set parameters
    playfield_h = history.shape[0]
    playfield_w = history.shape[1]
    decision_field = agent.decision_field
    pad_horizontal = np.zeros((playfield_h, decision_field - playfield_w, 10))
    pad_vertical = np.zeros((decision_field - playfield_h, decision_field, 10))
    # OHE 9 position means this cell is closed
    pad_horizontal[:, :, 9] = 1
    pad_vertical[:, :, 9] = 1
    # Get corner fields
    fields_for_predict = create_fields(history, pad_horizontal, pad_vertical)
    q_value = np.zeros((playfield_h, playfield_w))
    # Predict for top left corner
    tmp_q_value = agent.get_action(fields_for_predict[0].astype(np.float32))[0]
    q_value += tmp_q_value.squeeze()[0:playfield_h, 0:playfield_w]
    # Predict for top right corner
    tmp_q_value = agent.get_action(fields_for_predict[1].astype(np.float32))[0]
    q_value += tmp_q_value.squeeze()[0:playfield_h, -playfield_w:]
    # Predict for bottom left corner
    tmp_q_value = agent.get_action(fields_for_predict[2].astype(np.float32))[0]
    q_value += tmp_q_value.squeeze()[-playfield_h:, 0:playfield_w]
    # Predict for bottom right corner
    tmp_q_value = agent.get_action(fields_for_predict[3].astype(np.float32))[0]
    q_value += tmp_q_value.squeeze()[-playfield_h:, -playfield_w:]

    action = np.argmax(q_value)
    x = int(action // playfield_w)
    y = int(action - x * playfield_w)

    return x, y


def to_categorical(y, num_classes=None, dtype='float32'):
    # OHE presentation of input data
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)

    return categorical
