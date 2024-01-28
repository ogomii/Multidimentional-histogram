example usage of baseline.py:
when specifying number of bins for each dimenstion:
    python3 src_python\baseline.py --image images\nvidiaLogo.png --reduction_type columns --dims 4 4 4
when specifying bin ranges:
    python3 src_python\baseline.py --image images\nvidiaLogo.png --reduction_type columns --seq 0 63.75 127.5 191.25 255.0

Additionally, adding --save to the command will save counts in pythonCounts.txt for comparison with cuda output
You can use a linux diff command for comparison:
diff -w cudaCounts.txt pythonCounts.txt