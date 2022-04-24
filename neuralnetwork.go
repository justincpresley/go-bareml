package bareml

import "math"

type CostFunction uint8

const (
	COST_MSE CostFunction = 1
)

type NeuralNetwork struct {
	topologySize         int
	hiddenActivationType ActivationType
	outputActivationType ActivationType
	costFunction         CostFunction

	topology         []int
	layers           []*Layer
	weightMatrices   []*Matrix
	gradientMatrices []*Matrix

	input         []float64
	target        []float64
	errors        []float64
	derivedErrors []float64

	error        float64
	bias         float64
	momentum     float64
	learningRate float64
}

func NewNeuralNetwork(topology []int) *NeuralNetwork {
	nn := new(NeuralNetwork)
	nn.topology = topology
	nn.topologySize = len(topology)
	nn.hiddenActivationType = RELU
	nn.outputActivationType = SIGM
	nn.costFunction = COST_MSE
	nn.errors = make([]float64, topology[nn.topologySize-1])
	nn.derivedErrors = make([]float64, topology[nn.topologySize-1])
	nn.layers = make([]*Layer, nn.topologySize)
	nn.weightMatrices = make([]*Matrix, nn.topologySize-1)
	// initalize the topology
	for i := 0; i < nn.topologySize; i++ {
		if i > 0 && i < (nn.topologySize-1) {
			nn.layers[i] = NewLayer(topology[i], nn.hiddenActivationType)
		} else if i == (nn.topologySize - 1) {
			nn.layers[i] = NewLayer(topology[i], nn.outputActivationType)
		} else {
			nn.layers[i] = NewLayer(topology[i], 0)
		}
	}
	// initlize the weights
	for i := 0; i < (nn.topologySize - 1); i++ {
		nn.weightMatrices[i] = NewMatrix(topology[i], topology[i+1], true)
	}
	return nn
}

func (nn *NeuralNetwork) Train(input []float64, target []float64, bias float64, learningRate float64, momentum float64) {
	nn.learningRate = learningRate
	nn.bias = bias
	nn.momentum = momentum
	nn.target = target
	nn.input = input
	for i := 0; i < len(input); i++ {
		nn.layers[0].Set(i, input[i])
	}
	nn.feedForward()
	nn.setErrors()
	nn.backPropagation()
}

func (nn *NeuralNetwork) feedForward() {
	var (
		a *Matrix
		b *Matrix
		c *Matrix
	)
	for i := 0; i < (nn.topologySize - 1); i++ {
		if i != 0 {
			a = nn.ActivatedNeuronMatrix(i)
		} else {
			a = nn.RawNeuronMatrix(i)
		}
		b = nn.WeightMatrix(i)
		c = MultiplyMatrix(a, b)
		for c_index := 0; c_index < c.NumCols(); c_index++ {
			nn.setNeuronVal(i+1, c_index, c.Get(0, c_index)+nn.bias)
		}
	}
}

func (nn *NeuralNetwork) backPropagation() {
	var (
		newWeights       []*Matrix = make([]*Matrix, 0)
		deltaWeights     *Matrix
		gradients        *Matrix
		derivedValues    *Matrix
		zActivatedValues *Matrix
		tempNewWeights   *Matrix
		hiddenDerived    *Matrix
		indexOutputLayer = nn.topologySize - 1
		originalValue    float64
		deltaValue       float64
		g                float64
	)

	// PART 1 : OUTPUT to LAST HIDDEN LAYER
	gradients = NewMatrix(1, nn.topology[indexOutputLayer], false)
	derivedValues = nn.layers[indexOutputLayer].MatrixifyDerivedVals()
	for i := 0; i < nn.topology[indexOutputLayer]; i++ {
		gradients.Set(0, i, (nn.derivedErrors[i] * derivedValues.Get(0, i)))
	}

	// Gt * Z
	zActivatedValues = nn.layers[indexOutputLayer-1].MatrixifyActivatedVals()
	deltaWeights = MultiplyMatrix(gradients.Transpose(), zActivatedValues)

	// Compute for new weights (last hidden <-> output)
	tempNewWeights = NewMatrix(nn.topology[indexOutputLayer-1], nn.topology[indexOutputLayer], false)
	for r := 0; r < nn.topology[indexOutputLayer-1]; r++ {
		for c := 0; c < nn.topology[indexOutputLayer]; c++ {
			originalValue = nn.weightMatrices[indexOutputLayer-1].Get(r, c) * nn.momentum
			deltaValue = deltaWeights.Get(c, r) * nn.learningRate
			tempNewWeights.Set(r, c, (originalValue - deltaValue))
		}
	}
	newWeights = append(newWeights, tempNewWeights)

	// PART 2 : LAST HIDDEN LAYER to INPUT LAYER
	for i := (indexOutputLayer - 1); i > 0; i-- {
		gradients = MultiplyMatrix(gradients, nn.weightMatrices[i].Transpose())
		hiddenDerived = nn.layers[i].MatrixifyDerivedVals()

		for colCounter := 0; colCounter < hiddenDerived.NumCols(); colCounter++ {
			g = gradients.Get(0, colCounter) * hiddenDerived.Get(0, colCounter)
			gradients.Set(0, colCounter, g)
		}

		if i == 1 {
			zActivatedValues = nn.layers[0].MatrixifyRawVals()
		} else {
			zActivatedValues = nn.layers[i-1].MatrixifyActivatedVals()
		}

		deltaWeights = MultiplyMatrix(zActivatedValues.Transpose(), gradients)

		// update weights
		tempNewWeights = NewMatrix(nn.weightMatrices[i-1].NumRows(), nn.weightMatrices[i-1].NumCols(), false)
		for r := 0; r < tempNewWeights.NumRows(); r++ {
			for c := 0; c < tempNewWeights.NumCols(); c++ {
				originalValue = nn.weightMatrices[i-1].Get(r, c) * nn.momentum
				deltaValue = deltaWeights.Get(r, c) * nn.learningRate
				tempNewWeights.Set(r, c, (originalValue - deltaValue))
			}
		}
		newWeights = append(newWeights, tempNewWeights)
	}

	nn.weightMatrices = make([]*Matrix, len(newWeights))
	for i := 0; i < len(newWeights); i++ {
		nn.weightMatrices[i] = newWeights[len(newWeights)-i-1]
	}
}

func (nn *NeuralNetwork) setErrors() {
	switch nn.costFunction {
	case COST_MSE:
		nn.setErrorMSE()
		break
	default:
		nn.setErrorMSE()
		break
	}
}

func (nn *NeuralNetwork) setErrorMSE() {
	var (
		outputLayerIndex = len(nn.layers) - 1
		outputNeurons    = nn.layers[outputLayerIndex].GetNeurons()
		t                float64
		y                float64
	)
	nn.error = 0.00
	for i := 0; i < len(nn.target); i++ {
		t = nn.target[i]
		y = outputNeurons[i].ActivatedVal()
		nn.errors[i] = 0.5 * math.Pow(math.Abs(t-y), 2)
		nn.derivedErrors[i] = (y - t)
		nn.error += nn.errors[i]
	}
}

func (nn *NeuralNetwork) setNeuronVal(indexLayer int, indexNeuron int, val float64) {
	nn.layers[indexLayer].Set(indexNeuron, val)
}

func (nn *NeuralNetwork) ActivatedVals(index int) []float64 {
	return nn.layers[index].ActivatedVals()
}
func (nn *NeuralNetwork) RawVals(index int) []float64 {
	return nn.layers[index].RawVals()
}
func (nn *NeuralNetwork) DerivedVals(index int) []float64 {
	return nn.layers[index].DerivedVals()
}

func (nn *NeuralNetwork) RawNeuronMatrix(index int) *Matrix {
	return nn.layers[index].MatrixifyRawVals()
}
func (nn *NeuralNetwork) ActivatedNeuronMatrix(index int) *Matrix {
	return nn.layers[index].MatrixifyActivatedVals()
}
func (nn *NeuralNetwork) DerivedNeuronMatrix(index int) *Matrix {
	return nn.layers[index].MatrixifyDerivedVals()
}
func (nn *NeuralNetwork) WeightMatrix(index int) *Matrix {
	return nn.weightMatrices[index].Copy()
}
func (nn *NeuralNetwork) TotalError() float64 {
	return nn.error
}
