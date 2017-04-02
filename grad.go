package grad

import (
	"github.com/gonum/matrix/mat64"
)

type Grad struct {
	Weight    []float64
	Intercept float64
	Eta       float64
	Errors    []float64
}

func NewGrad(eta float64) *Grad {
	g := Grad{[]float64{}, 0.0, eta, []float64{}}
	g.Weight = []float64{RandNormal(0, 1), RandNormal(0, 1)}
	g.Intercept = RandNormal(0, 1)
	return &g
}

// objective function (calc conditional-prob): p(y|x) = 1 / 1 + exp(-y w x)
func (g *Grad) CalcLikeLihood(x *mat64.Dense, s int) (yhat []float64) {
	m := MulMulti(x, g.Weight, s) // like numpy.dot()

	for i := 0; i < len(m); i++ {
		yhat = append(yhat, Sigmoid(m[i]+g.Intercept))
	}
	return yhat
}

func (g *Grad) Grad(x *mat64.Dense, y []float64, s int) (w_grad, b_grad []float64) {
	errs := []float64{}

	yhat := g.CalcLikeLihood(x, s)

	for i := 0; i < len(y); i++ {
		errs = append(errs, y[i]-yhat[i]) // error = label - pred
	}
	e := mat64.NewVector(s, errs)

	w_grad = append(w_grad, -1*mat64.Dot(x.ColView(0), e))
	w_grad = append(w_grad, -1*mat64.Dot(x.ColView(1), e))
	b_grad = append(b_grad, -1*Mean(errs))

	return w_grad, b_grad
}
