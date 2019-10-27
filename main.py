import os
import numpy as np 
import utilities as utils 
import params 

from straightening_hierarchy import Straightening_Hierarchy
from torch.autograd import Variable

import scipy.io as sio

sequenceTypes = ['groundtruth', 'pixelfade', 'contrast']
modelStages = ['pixel', 'retina', 'v1']

savedir = '/scratch/ojh221/results/untangling/global/paper/figure6/'
savedir = '%s%s-N%d-K%d-S%d/' % (savedir, params.model_name, params.N, params.K, params.imgSize)
os.makedirs( savedir, exist_ok=True )

model = Straightening_Hierarchy(params.imgSize, N=params.N, K=params.K)
model.cuda()

for s, sequenceType in enumerate(sequenceTypes):

	curvature = {}
	for modelStage in modelStages:
		curvature[modelStage] = {} 

	if s == 2:
		imgNames = [ 'water-contrast0.5', 'prairieTer-contrastLog0.1', 'boats_contrastCocktail', 'bees_contrastCocktail', 'walking_contrastCocktail', 'egomotion_contrastCocktail', 'smile-contrastLog0.1', 'walking-contrast0.5', 'bees-contrast0.5', 'walking-contrastLog0.1' ]
	else:
		imgNames = [ 'water', 'prairieTer', 'boats', 'ice3', 'dogville', 'egomotion', 'walking', 'smile', 'bees', 'leaves-wind', 'carnegie-dam', 'chironomus' ]

	for i, imgName in enumerate(imgNames):

		if   s == 0:
			x = utils.makeGroundtruth(imgName) 
		elif s == 1:
			x = utils.makePixelfade(imgName) 
		elif s == 2:
			x = utils.makeContrastfade(imgName) 

		x = Variable( x.cuda() ) 
		y = model( x )

		for modelStage in modelStages:
			dY, cY = utils.computeDistCurv( y[modelStage] )
			curvature[modelStage][imgName] = cY.data.mean()

	for o, modelStage in enumerate(modelStages):	
		avPixelCurvature = np.array(list(curvature['pixel'].values())).mean()
		avModelCurvature = np.array(list(curvature[modelStage].values())).mean()
		deltaCurvature = avModelCurvature - avPixelCurvature

		print('sequence type: %s \tmodel stage: %s \tdelta curvature %.2f' % (sequenceType, modelStage, deltaCurvature))

	sio.savemat( savedir + 'modelCurvature_' + sequenceType + '.mat', curvature )