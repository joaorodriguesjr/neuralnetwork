import { iterate } from './helpers'

export class Neuron {
  public inputs: number[]

  public output: number

  public gradient: number

  public constructor(public weights: number[], public bias: number) {}

  public activate(inputs: number[]): number {
    this.inputs = inputs
    let weightedSum = inputs.reduce((total, input, index) => total + input * this.weights[index], this.bias)

    return this.output = this.activation(weightedSum)
  }

  public updateWeights(learningRate: number): void {
    this.weights = this.weights.map((weight, index) => weight - this.inputs[index] * learningRate * this.gradient)
    this.bias = this.bias - learningRate * this.gradient
  }

  public calculateOutputGradient(target: number): void {
    this.gradient = (this.output - target) * this.activationDerivative(this.output)
  }

  public calculateHiddenGradient(index: number, subsequentLayer: Neuron[]): void {
    // Compute the weighted sum of gradients from the subsequent layer
    const weightedSumOfGradients = subsequentLayer.reduce((sum, neuron) => {
      return sum + neuron.weights[index] * neuron.gradient;
    }, 0)

    // Compute the gradient for the current neuron in the hidden layer
    this.gradient = this.output * (1 - this.output) * weightedSumOfGradients
  }

  private activation(x: number): number {
    return 1 / (1 + Math.exp(-x))
  }

  private activationDerivative(x: number): number {
    return x * (1 - x)
  }

  public static create(inputsCount: number): Neuron {
    const weights = []
    iterate(inputsCount, () => weights.push(Math.random() * 2 - 1))

    return new Neuron(weights, Math.random() * 2 - 1)
  }
}
