example usage of baseline.py:
when specifying number of bins for each dimenstion:
    python3 src_python\baseline.py --image images\nvidiaLogo.png --reduction_type columns --dims 4 4 4
when specifying bin ranges:
    python3 src_python\baseline.py --image images\nvidiaLogo.png --reduction_type columns --ranges 0 63 127 191 255