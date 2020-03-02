import numpy as np
import argparse
parser = argparse.ArgumentParser(prog = 'Concatenate data')
parser.add_argument('--max_index', type=int, default=0)
parser.add_argument('--input_fstring', type=str, default="{}")
parser.add_argument('--output_filepath', type=str, default="untitled.npy")
parser.add_argument('--concatenate_axis', type=int, default=-1)
inputs = parser.parse_args()


out = []
for i in range(inputs.max_index):
    inp = np.load(inputs.input_fstring.format(i))
    out.append(inp)

out = np.array(out, dtype=np.float32)
if inputs.concatenate_axis != 0:
    out = np.moveaxis(out, 0, inputs.concatenate_axis)

np.save(f"{inputs.output_filepath}", out)