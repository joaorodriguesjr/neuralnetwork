import { Neuron } from './Neuron'
import { iterate } from './helpers'

export class Layer {
  public constructor(public neurons: Neuron[]) {}

  public process(inputs: number[]): number[] {
    return this.neurons.map((neuron) => neuron.activate(inputs))
  }

  public calculateOutputGradients(target: number[]): void {
    this.neurons.forEach((neuron, index) => neuron.calculateOutputGradient(target[index]))
  }

  public calculateHiddenGradients(subsequentLayer: Layer): void {
    this.neurons.forEach((neuron, index) => neuron.calculateHiddenGradient(index, subsequentLayer.neurons))
  }

  public updateWeights(learningRate: number): void {
    this.neurons.forEach((neuron) => neuron.updateWeights(learningRate))
  }

  public static create(neuronsCount: number, inputsCount: number): Layer {
    const neurons = []
    iterate(neuronsCount, () => neurons.push(Neuron.create(inputsCount)))

    return new Layer(neurons)

  }
}
