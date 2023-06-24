import { Layer } from './Layer'

export class Network {
  public constructor(public layers: Layer[]) {}

  public train(inputs: number[], targets: number[], learningRate: number): void {
    const outputs = this.predict(inputs)

    this.layers.reverse().forEach((layer, index) => {
      if (index === 0)
        layer.calculateOutputGradients(targets)
      else
        layer.calculateHiddenGradients(this.layers[index - 1])
    })

    this.layers.reverse().forEach((layer) => layer.updateWeights(learningRate))
  }

  public predict(inputs: number[]): number[] {
    return this.layers.reduce((inputs, layer) => layer.process(inputs), inputs)
  }

  public static create(structure: number[][]): Network {
    const layers = structure.map(([inputsCount, neuronsCount]) => Layer.create(neuronsCount, inputsCount))
    return new Network(layers)
  }
}
