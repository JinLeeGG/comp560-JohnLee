# Same input as with_colon, but body queries cannot attend to the final ':' key.
#
# This tests whether T5 fails because it can use ':' as a fixed positional anchor, or
# simply because the sequence length is 7.

exec(open('config/with_colon.py').read())

out_dir = 'out_with_colon_masked'
mask_marker_key = 'last'
