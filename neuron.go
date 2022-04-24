package bareml

import "math"

type ActivationType uint8

const (
	TANH ActivationType = 1
	RELU ActivationType = 2
	SIGM ActivationType = 3
)

type Neuron struct {
	rawVal         float64
	activatedVal   float64
	derivedVal     float64
	activationType ActivationType
}

func NewNeuron(val float64, aType ActivationType) *Neuron {
	n := new(Neuron)
	n.activationType = aType
	n.Set(val)
	return n
}

func (n *Neuron) Set(val float64) {
	n.rawVal = val
	n.Activate()
	n.Derive()
}

func (n *Neuron) Copy() *Neuron {
	return &Neuron{
		rawVal:         n.RawVal(),
		activatedVal:   n.ActivatedVal(),
		derivedVal:     n.DerivedVal(),
		activationType: n.ActivationType(),
	}
}

func (n *Neuron) Activate() {
	if n.activationType == TANH {
		n.activatedVal = math.Tanh(n.rawVal)
	} else if n.activationType == RELU {
		if n.rawVal > 0 {
			n.activatedVal = n.rawVal
		} else {
			n.activatedVal = 0
		}
	} else if n.activationType == SIGM {
		n.activatedVal = n.rawVal / (1 + math.Abs(n.rawVal))
	} else {
		n.activatedVal = n.rawVal / (1 + math.Abs(n.rawVal))
	}
}

func (n *Neuron) Derive() {
	if n.activationType == TANH {
		n.derivedVal = (1.0 - (n.activatedVal * n.activatedVal))
	} else if n.activationType == RELU {
		if n.rawVal > 0 {
			n.derivedVal = 1
		} else {
			n.derivedVal = 0
		}
	} else if n.activationType == SIGM {
		n.derivedVal = n.activatedVal * (1 - n.activatedVal)
	} else {
		n.derivedVal = n.activatedVal * (1 - n.activatedVal)
	}
}

func (n *Neuron) RawVal() float64 {
	return n.rawVal
}
func (n *Neuron) ActivatedVal() float64 {
	return n.activatedVal
}
func (n *Neuron) DerivedVal() float64 {
	return n.derivedVal
}
func (n *Neuron) ActivationType() ActivationType {
	return n.activationType
}
