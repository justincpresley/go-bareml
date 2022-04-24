package bareml

type Layer struct {
	size           int
	activationType ActivationType
	neurons        []*Neuron
}

func NewLayer(size int, aType ActivationType) *Layer {
	l := new(Layer)
	l.activationType = aType
	l.size = size
	l.neurons = make([]*Neuron, size)
	for i := 0; i < size; i++ {
		l.neurons[i] = NewNeuron(0.0000000000000, aType)
	}
	return l
}

func (l *Layer) Copy() *Layer {
	nl := new(Layer)
	nl.size = l.Size()
	nl.activationType = l.ActivationType()
	nl.neurons = make([]*Neuron, nl.size)
	for i := 0; i < nl.size; i++ {
		nl.neurons[i] = l.GetNeurons()[i].Copy()
	}
	return nl
}

func (l *Layer) MatrixifyRawVals() *Matrix {
	m := NewMatrix(1, l.size, false)
	for i := 0; i < l.size; i++ {
		m.Set(0, i, l.neurons[i].RawVal())
	}
	return m
}
func (l *Layer) MatrixifyActivatedVals() *Matrix {
	m := NewMatrix(1, l.size, false)
	for i := 0; i < l.size; i++ {
		m.Set(0, i, l.neurons[i].ActivatedVal())
	}
	return m
}
func (l *Layer) MatrixifyDerivedVals() *Matrix {
	m := NewMatrix(1, l.size, false)
	for i := 0; i < l.size; i++ {
		m.Set(0, i, l.neurons[i].DerivedVal())
	}
	return m
}

func (l *Layer) RawVals() []float64 {
	ret := make([]float64, l.size)
	for i := 0; i < l.size; i++ {
		ret[i] = l.neurons[i].RawVal()
	}
	return ret
}
func (l *Layer) ActivatedVals() []float64 {
	ret := make([]float64, l.size)
	for i := 0; i < l.size; i++ {
		ret[i] = l.neurons[i].ActivatedVal()
	}
	return ret
}
func (l *Layer) DerivedVals() []float64 {
	ret := make([]float64, l.size)
	for i := 0; i < l.size; i++ {
		ret[i] = l.neurons[i].DerivedVal()
	}
	return ret
}

func (l *Layer) GetNeurons() []*Neuron {
	return l.neurons
}
func (l *Layer) SetNeurons(ns []*Neuron) {
	l.neurons = ns
	l.size = len(l.neurons)
}
func (l *Layer) Size() int {
	return l.size
}
func (l *Layer) Set(i int, val float64) {
	l.neurons[i].Set(val)
}
func (l *Layer) ActivationType() ActivationType {
	return l.activationType
}
