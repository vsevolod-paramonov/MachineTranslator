import os 
from logger import logwriter


def _clear_dir(path):

    if os.path.exists(path):

        file1 = path + '.model'
        file2 = path + '.vocab'

        os.remove(file1)
        os.remove(file2)

def _write_file(seq, pth):

    with open(pth, 'w', encoding='utf-8') as f:
        for txt in seq:
            f.write(txt + '\n')

def _display_example(logwriter, inp, pred, right):

    msg = f'\nInput: {inp} \n Predicted: {pred} \n Target {right} \n'

    logwriter._log_custom_message(msg)