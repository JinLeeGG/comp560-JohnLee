# Small even/odd distance measurement sanity experiment with ':' present in attention context.
#
# The classifier still pools only over body tokens; ':' is included only so relative
# PE can attend to a fixed final delimiter anchor.

exec(open('config/basic.py').read())

out_dir = 'out_with_colon'
input_mode = 'with_colon'
block_size = 7
