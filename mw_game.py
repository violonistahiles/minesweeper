import numpy as np
import PySimpleGUI as sg
import os
from time import gmtime, strftime
import imageio
import shutil
import cv2
from model.minesweeper import Minesweeper
from model.agent import Agent
from utils.agent_utils import *
from utils.io_utils import *


if __name__ == '__main__':

    sg.theme('DarkAmber')  # Add a little color to your windows

    # Set directories
    module_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(module_dir, 'field_images')
    checkpoint_dir = os.path.join(module_dir, os.path.join('checkpoint', 'mw.onnx'))
    images_dir = os.path.join(module_dir, 'images')
    result_dir = os.path.join(module_dir, 'results')
    decision_field = 35

    # Set global parameters
    rewards = {'lose_reward': -2.5,
               'win_reward':  2.5,
               'yolo_reward': 0.1,
               'rep_point_reward': -0.5,
               'open_point_reward': 2.2
               }

    pictures = ['open.png', '1.png', '2.png', '3.png', '4.png', '5.png', '6.png',
                '7.png', '8.png', 'close.png', 'mine.png', 'open_mine.png']

    agent = Agent(decision_field, checkpoint_dir)

    # Set initial parameters for minesweeper
    layout = start_layout()
    window = sg.Window('Minesweeper', layout)

    # Game loop
    while True:
        event, values = window.read()

        # Case of "exit" button
        if event in (sg.WIN_CLOSED, 'Exit'):
            break

        # Case of "Start game" button
        if event in ('grid'):
            window.close()

            # Flags of game state for player and neural net
            nn_lose = False
            nn_win = False
            player_lose = False
            player_win = False

            # Game parameters
            playfield_h = int(values['playfield_h'])
            playfield_w = int(values['playfield_w'])
            mines_count = int(values['mines_count'])
            field_indices = [str(i) for i in range(playfield_h * playfield_w)]
            # Minesweeper classes for player and neural net
            minesweeper_for_player = Minesweeper(mines_count, playfield_h, playfield_w, rewards)
            minesweeper_for_nn = Minesweeper(mines_count, playfield_h, playfield_w, rewards)
            # Create game fields
            grid_for_player = create_grid(playfield_h, playfield_w, 0, path)
            grid_for_nn = create_grid(playfield_h, playfield_w, playfield_w * playfield_h, path)
            layout = create_layout(mines_count, playfield_h, playfield_w, grid_for_player, grid_for_nn)
            window = sg.Window('Minesweeper', layout)

        # Case of player field button
        if event in field_indices:

            # x, y of pressed buttom
            action = int(event)
            x = int(action // playfield_w)
            y = int(action - x * playfield_w)

            # If minesweepers initialized after first step
            if len(minesweeper_for_player.free_point) > 0:

                # Neural net step
                if nn_win or nn_lose:
                    new_field_nn = minesweeper_for_nn.minesfield.flatten()
                else:
                    x_nn, y_nn = get_predict(agent, history)
                    _, reward_nn, done_nn = minesweeper_for_nn.step((x_nn, y_nn))
                    if reward_nn == rewards['win_reward']:
                        window['nn_field'].Update('Neural network --- WIN')
                        nn_win = True
                    elif reward_nn == rewards['lose_reward']:
                        window['nn_field'].Update('Neural network --- LOSE')
                        nn_lose = True
                    else:
                        history = to_categorical(minesweeper_for_nn.minesfield, 10)

                    new_field_nn = minesweeper_for_nn.minesfield.flatten()

                # Player step
                if player_win or player_lose:
                    new_field_player = minesweeper_for_player.minesfield.flatten()
                else:
                    _, reward, done = minesweeper_for_player.step((x, y))
                    if reward == rewards['win_reward']:
                        window['pl_field'].Update('Player --- WIN')
                        player_win = True
                    elif reward == rewards['lose_reward']:
                        window['pl_field'].Update('Player --- LOSE')
                        player_lose = True

                    new_field_player = minesweeper_for_player.minesfield.flatten()

            else:  # Initialize minesweepers
                minesweeper_for_player.initialize_game((x, y))
                new_field_player = minesweeper_for_player.minesfield.flatten()
                # Copy neural net minesweeper parameters from player minesweeper
                minesweeper_for_nn.free_point = minesweeper_for_player.free_point.copy()
                minesweeper_for_nn.minesfield = minesweeper_for_player.minesfield.copy()
                minesweeper_for_nn.mines_coord = minesweeper_for_player.mines_coord.copy()
                minesweeper_for_nn.fake_playfield = minesweeper_for_player.fake_playfield.copy()

                history = to_categorical(minesweeper_for_nn.minesfield, 10)
                new_field_nn = minesweeper_for_nn.minesfield.flatten()

            # Update button fields for each minesweeper
            for i, ind in enumerate(field_indices):
                window[ind].update(image_filename=os.path.join(path, pictures[new_field_player[i]]))

            for i, ind in enumerate(field_indices, start=playfield_w * playfield_h):
                window[str(i)].update(image_filename=os.path.join(path, pictures[new_field_nn[int(ind)]]))

        # Case of "Save results" button
        if event == 'save':
            save_player = values['save_pl']  # Save or not player results
            save_nn = values['save_nn']  # Save or not neural networks results
            nn_history = minesweeper_for_nn.history.copy()  # Current nn history
            player_history = minesweeper_for_player.history.copy() # Current player history
            nn_history_len = len(nn_history)
            player_history_len = len(player_history)
            # Prepare folder for images
            if os.path.exists(images_dir) is True:
                shutil.rmtree(images_dir)
            os.mkdir(images_dir)
            # Expand history with lower length
            if nn_history_len < player_history_len:
                frame_to_extend = minesweeper_for_nn.history[-1].copy()
                frames_to_add = player_history_len - nn_history_len
                nn_history.extend([frame_to_extend] * frames_to_add)
            elif nn_history_len > player_history_len:
                frame_to_extend = minesweeper_for_player.history[-1].copy()
                frames_to_add = nn_history_len - player_history_len
                player_history.extend([frame_to_extend] * frames_to_add)
            # Prepare step result for saving
            for i in range(len(player_history)):
                data_player = save_result(player_history[i], True)
                data_nn = save_result(nn_history[i], False)
                if save_player and save_nn:
                    data = np.concatenate((data_player, data_nn), axis=1)
                elif save_player:
                    data = data_player
                elif save_nn:
                    data = data_nn
                else:
                    continue

                cv2.imwrite(os.path.join(images_dir, str(i) + '.jpg'), data[:, :, ::-1])
            # Create .gif
            if len(os.listdir(images_dir)) > 0:
                str_time = strftime('%d%m_%H%M%S', gmtime())
                images = [imageio.imread(os.path.join(images_dir, str(id) + '.jpg')) for id in
                          range(len(os.listdir(images_dir)))]
                imageio.mimsave(os.path.join(result_dir, f'{str_time}_{playfield_h}x{playfield_w}x{mines_count}.gif'),
                                images)

        # Case of "Complete" button
        if event == 'complete':
            # While neural network dont lose or win, continue playing
            while nn_win == nn_lose:
                x_nn, y_nn = get_predict(agent, history)
                _, reward_nn, done_nn = minesweeper_for_nn.step((x_nn, y_nn))

                if reward_nn == rewards['win_reward']:
                    window['nn_field'].Update('Neural network --- WIN')
                    nn_win = True
                elif reward_nn == rewards['lose_reward']:
                    window['nn_field'].Update('Neural network --- LOSE')
                    nn_lose = True
                else:
                    history = to_categorical(minesweeper_for_nn.minesfield, 10)

                new_field_nn = minesweeper_for_nn.minesfield.flatten()

                for i, ind in enumerate(field_indices, start=playfield_w * playfield_h):
                    window[str(i)].update(image_filename=os.path.join(path, pictures[new_field_nn[int(ind)]]))

                window.Refresh()

    window.close()
