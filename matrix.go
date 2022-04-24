package bareml

import (
	"math/rand"
	"time"
)

func init() {
	rand.Seed(time.Now().UnixNano())
}

type Matrix struct {
	numRows int
	numCols int
	values  [][]float64
}

func NewMatrix(numRows int, numCols int, isRandom bool) *Matrix {
	m := new(Matrix)
	m.numRows = numRows
	m.numCols = numCols
	m.values = make([][]float64, numRows)
	if !isRandom {
		for i := 0; i < numRows; i++ {
			m.values[i] = make([]float64, numCols)
		}
	} else {
		for i := 0; i < numRows; i++ {
			m.values[i] = make([]float64, numCols)
			for j := 0; j < numCols; j++ {
				m.values[i][j] = rand.Float64()
			}
		}
	}
	return m
}

func (m *Matrix) Copy() *Matrix {
	c := NewMatrix(m.numRows, m.numCols, false)
	for i := 0; i < m.numRows; i++ {
		for j := 0; j < m.numCols; j++ {
			c.Set(i, j, m.Get(i, j))
		}
	}
	return c
}

func (m *Matrix) Transpose() *Matrix {
	t := NewMatrix(m.numCols, m.numRows, false)
	for i := 0; i < m.numRows; i++ {
		for j := 0; j < m.numCols; j++ {
			t.Set(j, i, m.Get(i, j))
		}
	}
	return t
}

func (m *Matrix) Set(row int, col int, val float64) {
	m.values[row][col] = val
}
func (m *Matrix) Get(row int, col int) float64 {
	return m.values[row][col]
}
func (m *Matrix) NumRows() int {
	return m.numRows
}
func (m *Matrix) NumCols() int {
	return m.numCols
}

func MultiplyMatrix(a *Matrix, b *Matrix) *Matrix {
	c := NewMatrix(a.NumRows(), b.NumCols(), false)
	for i := 0; i < a.NumRows(); i++ {
		for j := 0; j < b.NumCols(); j++ {
			for k := 0; k < b.NumRows(); k++ {
				c.Set(i, j, c.Get(i, j)+(a.Get(i, k)*b.Get(k, j)))
			}
			c.Set(i, j, c.Get(i, j))
		}
	}
	return c
}
