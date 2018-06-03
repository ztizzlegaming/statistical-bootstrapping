# MAT 311 - Mathematical Statistics
# Confidence Intervals for Skewness and Excess Kurtosis by Bootstrapping
# 
# This program can take a long time to run. I saved a bit of time by
# calculating xbar as I was resampling, or calculating sample skewness and
# kurtosis in the same loop, but as n or the number of bootstrap resamples
# becomes very large, it can take several minutes to hours to run.
# 
# To run this program:
# python3 bootstrap.py
# 
# For sampling, I use the numpy library. It can be installed with pip3:
# pip3 install numpy
# 
# Skewness and Excess Kurtosis for all of the distributions:
# normal: skewness = 0, excess kurtosis = 0
# uniform: skewness = 0, excess kurtosos = -6/5
# exponential: skewness = 2, excess kurtosis = 6
# erlang: skewness = 2 / sqrt(alpha), excess kurtosis: 6 / alpha
# binomial: skewness = (1 - 2p) / sqrt[np(1 - p)], excess kurtosos: [1 - 6p(1 - p)] / [np(1 - p)]
# poisson: lambda^(-1/2), excess kurtosos: lambda^(-1)
# 
# Jordan Turley

import random
import numpy
import math

# The number of initial data points to sample from the distribution
SAMPLES = 100

# The number of times to resample the data to generate the bootstrap resamples
BOOTSTRAP_RESAMPLES = 100

# The number of times to repeat the confidence interval
REPEATS = 100

# The level of alpha to generate the confidence interval for
ALPHA = 0.2
LOWER = math.floor((ALPHA / 2) * BOOTSTRAP_RESAMPLES)
UPPER = math.floor((1 - ALPHA / 2) * BOOTSTRAP_RESAMPLES)

# Distributions to sample from and their parameters
DISTRIBUTIONS = {
	'normal': [0, 1], # mean, standard deviation
	'uniform': [0, 1], # min, max
	'exponential': [1], # mean
	'erlang': [1, 1], # alpha, beta
	'binomial': [100, 0.5], # n, p
	'poisson': [2] # lambda
}

def main():
	# Loop over each distribution
	distributionNames = DISTRIBUTIONS.keys()
	for distribution in distributionNames:
		#print('Distribution:', distribution)
		#print('Parameters:', DISTRIBUTIONS[distribution])
		skewnessCorrect = 0
		excessKurtosisCorrect = 0
		for _ in range(REPEATS):
			# First calculate the real skewness and excess kurtosis of the distribution
			realSkewness = distributionSkewness(distribution, DISTRIBUTIONS[distribution])
			realExcessKurtosis = distributionExcessKurtosis(distribution, DISTRIBUTIONS[distribution])

			# Generate our sample
			s = sample(distribution, DISTRIBUTIONS[distribution], SAMPLES)
			xbar = numpy.mean(s)

			# Calculate the skewness of the original sample
			sSkewness, sExcessKurtosis = sampleStatistics(s, xbar)

			# Resample a certain number of times from the sample
			skewnessValues = []
			excessKurtosisValues = []
			for i1 in range(BOOTSTRAP_RESAMPLES):
				newS, newXBar = resample(s)

				# Calculate skewness and excess kurtosis of resample
				skewness, excessKurtosis = sampleStatistics(newS, newXBar)

				skewnessDifference = skewness - sSkewness
				skewnessValues.append(skewnessDifference)

				excessKurtosisDifference = excessKurtosis - sExcessKurtosis
				excessKurtosisValues.append(excessKurtosisDifference)

			# Sort all of our difference values
			skewnessValues.sort()
			excessKurtosisValues.sort()

			# Get the bounds for our skewness confidence interval
			lowerBoundSkewness = sSkewness - skewnessValues[UPPER]
			upperBoundSkewness = sSkewness - skewnessValues[LOWER]

			# Get the bounds for our excess kurtosis confidence interval
			lowerBoundExcessKurtosis = sExcessKurtosis - excessKurtosisValues[UPPER]
			upperBoundExcessKurtosis = sExcessKurtosis - excessKurtosisValues[LOWER]

			# print(realSkewness, ' Skewness CI: (', lowerBoundSkewness, ', ', upperBoundSkewness, ')', sep = '')
			# print(realExcessKurtosis, ' EK CI: (', lowerBoundExcessKurtosis, ', ', upperBoundExcessKurtosis, ')', sep = '')

			# Check if the real skewness value is in our bound
			if realSkewness > lowerBoundSkewness and realSkewness < upperBoundSkewness:
				skewnessCorrect += 1

			# Check if the real excess kurtosis value is in our bound
			if realExcessKurtosis > lowerBoundExcessKurtosis and realExcessKurtosis < upperBoundExcessKurtosis:
				excessKurtosisCorrect += 1

		print(distribution, 'Skewness:', skewnessCorrect, '/', REPEATS, 'Kurtosis:', excessKurtosisCorrect, '/', REPEATS)

	# Test how the width decreases as n increases
	nVals = [8, 16, 32, 64, 128, 256, 512, 1024];
	for n in nVals:
		averageSkewnessWidth = 0
		averageKurtosisWidth = 0
		for _ in range(REPEATS):
			s = sample('normal', [0, 1], n)
			xbar = numpy.mean(s)

			# Calculate the skewness of the original sample
			sSkewness, sExcessKurtosis = sampleStatistics(s, xbar)

			# Resample a certain number of times from the sample
			skewnessValues = []
			excessKurtosisValues = []
			for i1 in range(BOOTSTRAP_RESAMPLES):
				newS, newXBar = resample(s)

				# Calculate skewness and excess kurtosis of resample
				skewness, excessKurtosis = sampleStatistics(newS, newXBar)

				skewnessDifference = skewness - sSkewness
				skewnessValues.append(skewnessDifference)

				excessKurtosisDifference = excessKurtosis - sExcessKurtosis
				excessKurtosisValues.append(excessKurtosisDifference)

			# Sort all of our difference values
			skewnessValues.sort()
			excessKurtosisValues.sort()

			# Get the bounds for our skewness confidence interval
			lowerBoundSkewness = sSkewness - skewnessValues[UPPER]
			upperBoundSkewness = sSkewness - skewnessValues[LOWER]

			diff = upperBoundSkewness - lowerBoundSkewness
			averageSkewnessWidth += diff

			# Get the bounds for our excess kurtosis confidence interval
			lowerBoundExcessKurtosis = sExcessKurtosis - excessKurtosisValues[UPPER]
			upperBoundExcessKurtosis = sExcessKurtosis - excessKurtosisValues[LOWER]

			diff = upperBoundExcessKurtosis - lowerBoundExcessKurtosis
			averageKurtosisWidth += diff

		averageSkewnessWidth /= REPEATS
		averageKurtosisWidth /= REPEATS

		print('Skewness width with n = ', n, ': ', averageSkewnessWidth, sep = '')
		print('Excess Kurtosis width with n = ', n, ': ', averageKurtosisWidth, sep = '')

def sample(distribution, params, n):
	'''Generates a sample from a given distribution with given parameters, and
	sample size n'''
	if distribution == 'normal':
		return numpy.random.normal(params[0], params[1], n)
	elif distribution == 'uniform':
		return numpy.random.uniform(params[0], params[1], n)
	elif distribution == 'exponential':
		return numpy.random.exponential(params[0], n)
	elif distribution == 'erlang':
		return numpy.random.gamma(params[0], params[1], n)
	elif distribution == 'binomial':
		return numpy.random.binomial(params[0], params[1], n)
	elif distribution == 'poisson':
		return numpy.random.poisson(params[0], n)
	else:
		print('The distribution was not recognized')

def resample(sample):
	'''Performs the bootstrap resampling, returning the resample and the
	average of the resample, to save some computation later.'''
	newSample = []
	xSum = 0
	for i1 in range(len(sample)):
		randomIdx = random.randint(0, len(sample) - 1)
		newSample.append(sample[randomIdx])
		xSum += sample[randomIdx]

	# Calculate average of all points resampled
	xBar = xSum / len(sample)

	return newSample, xBar

def distributionSkewness(distribution, params):
	'''Gets the actual value for the skewness for a given distribution.
	These values/formulas were gotten from Wikipedia.'''
	if distribution == 'normal' or distribution == 'uniform':
		return 0
	elif distribution == 'exponential':
		return 2
	elif distribution == 'erlang':
		return 2 / math.sqrt(params[0])
	elif distribution == 'binomial':
		return (1 - 2 * params[1]) / math.sqrt(params[0] * params[1] * (1 - params[1]))
	elif distribution == 'poisson':
		return params[0] ** (-1/2)
	else:
		print('The distribution was not recognized')

def distributionExcessKurtosis(distribution, params):
	'''Gets the actual value for the excess kurtosis for a given distribution.
	These values/formulas were gotten from Wikipedia.'''
	if distribution == 'normal':
		return 0
	elif distribution == 'uniform':
		return -6 / 5
	elif distribution == 'exponential':
		return 6
	elif distribution == 'erlang':
		return 6 / params[0]
	elif distribution == 'binomial':
		return (1 - 6 * params[1] * (1 - params[1])) / (params[0] * params[1] * (1 - params[1]))
	elif distribution == 'poisson':
		return 1 / params[0]
	else:
		print('The distribution was not recognized')

def sampleStatistics(sample, xbar = None):
	'''Calculates the sample skewness and kurtosis from a given sample.
	Uses given xbar, or if none is given, calculates it.'''
	if xbar == None:
		xbar = numpy.mean(sample)

	numeratorSkewness = 0
	denominatorSkewness = 0

	numeratorExcessKurtosis = 0
	denominatorExcessKurtosis = 0

	# Loop over sample points and calculate the numerator and denominator sums
	for samplePoint in sample:
		vSum = (samplePoint - xbar) ** 2

		numeratorSkewness += (samplePoint - xbar) ** 3
		denominatorSkewness += vSum

		numeratorExcessKurtosis += (samplePoint - xbar) ** 4
		denominatorExcessKurtosis += vSum

	numeratorSkewness /= SAMPLES
	denominatorSkewness /= (SAMPLES - 1)
	denominatorSkewness **= (3 / 2)

	numeratorExcessKurtosis /= SAMPLES
	denominatorExcessKurtosis /= (SAMPLES - 1)
	denominatorExcessKurtosis **= 2

	skewness = 0
	if denominatorSkewness != 0:
		skewness = numeratorSkewness / denominatorSkewness

	excessKurtosis = 0
	if denominatorExcessKurtosis != 0:
		excessKurtosis = (numeratorExcessKurtosis / denominatorExcessKurtosis) - 3

	return skewness, excessKurtosis

if __name__ == '__main__':
	main()