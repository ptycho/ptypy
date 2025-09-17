import os
import numpy as np
import matplotlib.pyplot as plt
import ptypy

# if output directory does not exist, create it
if not os.path.exists("./userguide/generated/"):
	os.mkdir("./userguide/generated/")

def create_test_image(outdir="./userguide/generated/", outfile="test.png"):
	plt.figure()
	plt.scatter(np.random.random(10), np.random.random(10), c='r')
	plt.tight_layout()
	plt.savefig(f'{outdir}{outfile}')

