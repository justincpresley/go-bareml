# A Bare-bones Machine Learning Library in Go

This Go library provides an API which allows for basic machine learning capability.
This library contains no external dependencies, is completely go-based, and fully implements a neural network.

*I followed along with a C++ version while making this library. You can links to that version below.*

Links: [Repository](https://github.com/ralampay/ann) - [Videos](https://www.youtube.com/watch?v=PQo78WNGiow&list=PL2-7U6BzddIYBOl98DDsmpXiTcj1ojgJG)


## Documentation / Usage

The full API documentation is available on [pkg.go.dev](https://pkg.go.dev/github.com/justincpresley/go-bareml).

## Installation

```
go get -u github.com/justincpresley/go-bareml
```

Or use your favorite golang vendoring tool.

## Example

```go
package main

import (
	"fmt"

	"github.com/justincpresley/go-bareml"
)

func main() {
	var input = []float64{0.2, 0.5, 0.1}
	var target = []float64{0.2, 0.5}

	// The first int must be len(input) and the last int must be len(target)
	var topology = []int{3, 2, 10, 2}

	var (
		learningRate float64 = 0.05
		momentum     float64 = 1
		bias         float64 = 1
	)

	neuralnet := bareml.NewNeuralNetwork(topology)

	fmt.Printf("Begin Training...\n")
	for i := 0; i < 1000; i++ {
		neuralnet.Train(input, target, bias, learningRate, momentum)
		fmt.Printf("Epoch: %d | Error %f\n", i, neuralnet.TotalError())
	}
}
```

## Notes

This was a learning project for me to get comfortable with Go. I originally aimed to improve the design and add additional features / models. However, I am now unable to maintain this library due to more pressing work in networking and cybersecurity. Feel free to fork the repository and build off it though!