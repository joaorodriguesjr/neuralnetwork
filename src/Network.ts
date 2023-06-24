import { Layer } from './Layer'

export class Network {
  public constructor(public layers: Layer[]) {}

  public predict(inputs: number[]): number[] {
    return this.layers.reduce((inputs, layer) => layer.process(inputs), inputs)
  }

  public train(inputs: number[], targets: number[], learningRate: number): void {
    this.predict(inputs)
    this.calculateGradients(targets)

    this.layers.forEach((layer) => layer.updateWeights(learningRate))
  }

  private calculateGradients(targets: number[]): void {
    this.layers.reverse()

    this.layers.forEach((layer, index) => {
      (! index)
        ? layer.calculateOutputGradients(targets)
        : layer.calculateHiddenGradients(this.layers[index - 1])
    })

    this.layers.reverse()
  }

  public static create(structure: number[][]): Network {
    const layers = structure.map(([inputsCount, neuronsCount]) => Layer.create(neuronsCount, inputsCount))
    return new Network(layers)
  }
}
