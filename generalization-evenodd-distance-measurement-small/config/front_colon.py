# Small even/odd distance measurement sanity experiment with ':' at the front.
#
# The classifier still pools only over body tokens; ':' is included only so relative
# PE can attend to a fixed delimiter anchor at the opposite side of the sequence.

exec(open('config/basic.py').read())

out_dir = 'out_front_colon'
input_mode = 'front_colon'
block_size = 7
