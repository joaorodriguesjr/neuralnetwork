import { iterate } from './helpers'

export class Neuron {
  public output: number

  public inputs: number[]

  public gradient: number

  public constructor(public weights: number[], public bias: number) {}

  public activate(inputs: number[]): number {
    let weightedSum = inputs.reduce((total, input, index) => total + input * this.weights[index], this.bias)

    this.inputs = inputs
    this.output = this.activation(weightedSum)

    return this.output
  }

  public updateWeights(learningRate: number): void {
    this.weights = this.weights.map((weight, index) => weight - this.inputs[index] * learningRate * this.gradient)
    this.bias -= this.gradient * learningRate
  }

  public calculateOutputGradient(target: number): void {
    this.gradient = (this.output - target) * this.activationDerivative(this.output)
  }

  public calculateHiddenGradient(index: number, subsequentLayer: Neuron[]): void {
    const weightedSumOfGradients = subsequentLayer.reduce((sum, neuron) => {
      return sum + neuron.weights[index] * neuron.gradient;
    }, 0)

    this.gradient = this.activationDerivative(this.output) * weightedSumOfGradients
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
