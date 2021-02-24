import numpy as np
import matplotlib.pyplot as plt
import PySimpleGUI as sg
import os


def save_result(history, player):
    '''
    Draw current play field from history
    Input:
    - history - play field from history
    - player - flag, if true means history of player, else history of neural net
    Output:
    - data - ndarray of figure representing current step
    '''

    # Colors for digits
    text_color = ['black', 'blue', 'green', 'red', 'red', 'red', 'red', 'red', 'red']
    # Colors for play field cells
    cell_color = [[100, 100, 100], [150, 150, 150], [0, 0, 0]]
    # Pixel size of cell
    step = 20

    # Field for results
    numpy_grid = np.zeros((history.shape[0] * step, history.shape[1] * step, 3))

    # Paint cells
    for x in range(history.shape[0]):
        for y in range(history.shape[1]):
            if history[x, y] != 9:  # If cell open
                numpy_grid[x * step:x * step + step, y * step:y * step + step] = cell_color[1]
            else:  # If cell closed
                numpy_grid[x * step:x * step + step, y * step:y * step + step] = cell_color[0]

    # Paint black separation lines
    for x in range(0, numpy_grid.shape[0], step):
        for y in range(0, numpy_grid.shape[1], step):
            numpy_grid[x:x + 1, y:y + step] = cell_color[2]
            numpy_grid[x + step - 1:x + step, y:y + step] = cell_color[2]
            numpy_grid[x:x + step, y:y + 1] = cell_color[2]
            numpy_grid[x:x + step, y + step - 1:y + step] = cell_color[2]

    fig = plt.figure(figsize=(history.shape[1] // 2, history.shape[0] // 2))
    plt.imshow(numpy_grid / 255)
    plt.axis('off')

    # Place digits
    for x, xn in enumerate(range(step // 2, numpy_grid.shape[0], step)):
        for y, yn in enumerate(range(step // 2, numpy_grid.shape[1], step)):
            if history[x, y] != 9:
                if history[x, y] in [10, 11]:
                    label = '*'
                    if history[x, y] == 11:
                        plt.text(yn - 1, xn + 4, label, color='red', ha='center', va='center', fontsize=25)
                    else:
                        plt.text(yn - 1, xn + 4, label, color='black', ha='center', va='center', fontsize=25)
                elif history[x, y] < 9 and history[x, y] > 0:
                    label = history[x, y]
                    plt.text(yn, xn, label, color=text_color[label], ha='center', va='center', fontsize=23)

    # Plot title
    if player:
        plt.title('Player', fontdict={'fontsize': 20})
    else:
        plt.title('Neural network', fontdict={'fontsize': 20})

    # Plot figure
    fig.canvas.draw()
    # Save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close()
    return data


def create_grid(height, width, start, path):
    # Create grid of buttons for minesweeper filed
    # start - start serial number for button keys in current field
    # path - absolute path for button pictures
    grid_layout = []
    for r in range(height):
        row = []
        for c in range(width):
            t = sg.Button(size=(2, 1), pad=(0, 0), image_filename=os.path.join(path, 'close.png'),
                          key=str(start + width * r + c))
            # add to tile array
            row.append(t)
        grid_layout.append(row)
    grid = sg.Column(grid_layout)

    return grid


def start_layout():
    # Start layout
    layout = [[sg.Text('Field height'), sg.InputText('8', key='playfield_h', size=(5, 1)),
               sg.Text('Field width'), sg.InputText('8', key='playfield_w', size=(5, 1)),
               sg.Text('Mines count'), sg.InputText('10', key='mines_count', size=(5, 1))],
              [sg.HorizontalSeparator()],
              [sg.Column([[sg.Button(button_text='Start game', key='grid')]]),
               sg.Column([[sg.Cancel('Exit')]], element_justification='right', expand_x=True)]
              ]

    return layout


def create_layout(mines_count, playfield_h, playfield_w, grid_for_player, grid_for_nn):
    # Layout for next games
    layout = [[sg.Text('Field height'), sg.InputText(f'{playfield_h}', key='playfield_h', size=(5, 1)),
               sg.Text('Field width'), sg.InputText(f'{playfield_w}', key='playfield_w', size=(5, 1)),
               sg.Text('Mines count'), sg.InputText(f'{mines_count}', key='mines_count', size=(5, 1))],
              [sg.HorizontalSeparator()],
              [sg.Column([[sg.Text('Player', key='pl_field', size=(25, 1))], [grid_for_player],
                          [sg.Checkbox('Save player results', default=True, key='save_pl')]]),
               sg.HorizontalSeparator(),
               sg.Column([[sg.Text('Neural network', key='nn_field', size=(25, 1))], [grid_for_nn],
                          [sg.Checkbox('Save NNet results', default=True, key='save_nn')]])],
              [sg.HorizontalSeparator()],
              [sg.Column([[sg.Button(button_text='Start game', key='grid'),
                           sg.Button(button_text='Save results', key='save'),
                           sg.Button(button_text='Complete', key='complete')]]),
               sg.Column([[sg.Cancel('Exit')]], element_justification='right', expand_x=True)],
              [sg.Text('* Results folder - minesweeper_directory/results/')],
              [sg.Text('** If You losed before neural net, just press "Complete" button')]
              ]

    return layout

