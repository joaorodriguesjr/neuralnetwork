export function iterate(count: number, callback: (index: number) => void): void {
  for (let index = 0; index < count; index++) callback(index)
}
