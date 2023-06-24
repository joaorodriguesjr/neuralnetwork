import { Network } from './Network'
import { iterate } from './helpers'

const network = Network.create([
  [2, 2], [2, 2], [2, 1]
])

iterate(100000, () => {
  network.train([0, 0], [0], 0.1)
  network.train([1, 1], [0], 0.1)
  network.train([0, 1], [1], 0.1)
  network.train([1, 0], [1], 0.1)
})

console.log(
  Math.round(network.predict([0, 0])[0]),
  Math.round(network.predict([1, 1])[0]),
  Math.round(network.predict([1, 0])[0]),
  Math.round(network.predict([0, 1])[0])
)
