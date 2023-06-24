import { readFileSync, writeFileSync } from 'fs'
import { Layer } from './Layer'
import { Neuron } from './Neuron'
import { Network } from './Network'

export function store(network: Network): void {
  const data = network.layers.map(layer => layer.neurons.map(neuron => ({ weights: neuron.weights, bias: neuron.bias })))
  writeFileSync('data/trained.json', JSON.stringify(data, null, 2))
}

export function restore(): Network {
  const data = JSON.parse(readFileSync('data/trained.json').toString())
  const layers = data.map(layer => new Layer(layer.map(neuron => new Neuron(neuron.weights, neuron.bias))))
  return new Network(layers)
}

export function iterate(count: number, callback: (index: number) => void): void {
  for (let index = 0; index < count; index++) callback(index)
}
