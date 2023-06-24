import { Neuron } from '../src/Neuron'

describe('Neuron', () => {
  describe('activate', () => {
    it('should correctly calculate the output value', () => {
      const neuron = new Neuron([0.5, 0.5], 0.0)
      const inputs = [1, 1]
      const expectedOutput = 1 / (1 + Math.exp(-1.0))

      const output = neuron.activate(inputs)

      expect(output).toBeCloseTo(expectedOutput)
    })

    it('should update the inputs property with the provided inputs', () => {
      const neuron = new Neuron([0.5, 0.5], 0)
      const inputs = [1, 0]

      neuron.activate(inputs)

      expect(neuron.inputs).toEqual(inputs)
    })
  })

  describe('updateWeights', () => {
    it('should correctly update the weights and bias based on the gradient and learning rate', () => {
      const neuron = new Neuron([0.5, 0.5], 0)
      neuron.inputs = [1, 0]
      neuron.gradient = 0.1
      const learningRate = 0.2
      const expectedWeights = [0.5 - 1 * learningRate * 0.1, 0.5 - 0 * learningRate * 0.1]
      const expectedBias = 0 - learningRate * 0.1

      neuron.updateWeights(learningRate)

      expect(neuron.weights).toEqual(expectedWeights)
      expect(neuron.bias).toBeCloseTo(expectedBias)
    })
  })

  describe('calculateOutputGradient', () => {
    it('should correctly calculate the gradient for an output neuron', () => {
      const neuron = new Neuron([0.5, 0.5], 0)
      neuron.output = 0.8
      const target = 0.2
      const expectedGradient = (0.8 - 0.2) * (0.8 * (1 - 0.8))

      neuron.calculateOutputGradient(target)

      expect(neuron.gradient).toBeCloseTo(expectedGradient)
    })
  })

  describe('calculateHiddenGradient', () => {
    it('should correctly calculate the gradient for a hidden neuron', () => {
      const neuron = new Neuron([0.2, 0.3], 0)
      neuron.output = 0.5

      const subsequentNeuron = new Neuron([0.5], 0)
      subsequentNeuron.gradient = 0.5

      const expectedWeightedSum = (0.2 * 0.5) + (0.3 * 0.5)
      const expectedGradient = expectedWeightedSum * (0.5 * (1 - 0.5))

      neuron.calculateHiddenGradient(0, [subsequentNeuron])

      expect(neuron.gradient).toBeCloseTo(expectedGradient)
    })
  })
})
