import matplotlib.pyplot as plt
import numpy as np

loaddir = 'results/retina-complexV1-N6-K8-S256/'
sequenceTypes = ['groundtruth', 'pixelfade', 'contrast']
xaxis = range(3)

for sequence in sequenceTypes:

	loadfile = '%smodelCurvature_%s.npy' % (loaddir, sequence)
	x = np.load(loadfile, allow_pickle=True).item()
	y = []

	for layer, values in x.items():

		values = np.array(list(values.values()))
		mean = values.mean()
		y.append(mean)

	y = np.array(y) - y[0]
	print(sequence, y)

	plt.plot(xaxis, y, '-o', label=sequence)

plt.xticks(ticks=xaxis, labels=['Pixels', 'LGN', 'V1'])
plt.legend()
plt.show()


