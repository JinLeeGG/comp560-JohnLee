# Tiny relative-order sanity experiment with ':' present in attention context.
#
# The classifier still pools only over body tokens; ':' is included only so relative
# PE can attend to a fixed final delimiter anchor.

exec(open('config/basic.py').read())

out_dir = 'out_body_plus_colon'
input_mode = 'body_plus_colon'
block_size = 7
