function iterate(count: number, callback: (index: number) => void): void {
  for (let index = 0; index < count; index++) callback(index)
}

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

const network = Network.create([
  [2, 2], [2, 2], [2, 1]
])

iterate(100000, () => {
  network.train([0, 0], [0], 0.1)
  network.train([0, 1], [1], 0.1)
  network.train([1, 0], [1], 0.1)
  network.train([1, 1], [0], 0.1)
})

iterate(0, () => {
})

iterate(0, () => {
})

iterate(0, () => {
})

console.log(
  Math.round(network.predict([0, 0])[0]),
  Math.round(network.predict([1, 1])[0]),
  Math.round(network.predict([1, 0])[0]),
  Math.round(network.predict([0, 1])[0])
)
