package grad

import (
	"math"

	"github.com/gonum/matrix/mat64"
)

// Loss function
func sigmoid(x float64) (y float64) {
	return 1.0 / (1.0 + math.Exp(-1*x))
}

func unSigmoid(x float64) (y float64) {
	x = x*0.99 + 0.01
	return math.Log(x / (1 - x))
}

// objective function (likelihood) : p(y|x) = 1 / 1 + exp(-y w x)
func P_y_given_x(x *mat64.Dense, w []float64, b float64, s int) (l []float64) {
	m := mulMulti(x, w, s)

	z := []float64{}
	for i := 0; i < len(m); i++ {
		z = append(z, m[i]+b)
	}

	// Sum
	for i := 0; i < len(z); i++ {
		l = append(l, sigmoid(z[i]))
	}
	return l
}

func Grad(x *mat64.Dense, c, w []float64, b float64, s int) (w_grad, b_grad []float64) {
	errs := []float64{}

	l := P_y_given_x(x, w, b, s)

	for i := 0; i < len(c); i++ {
		errs = append(errs, c[i]-l[i]) // error = label - loss
	}
	e := mat64.NewDense(s, 1, errs)

	w_grad = append(w_grad, -1*mat64.Dot(x.ColView(0), e))
	w_grad = append(w_grad, -1*mat64.Dot(x.ColView(1), e))
	b_grad = append(b_grad, -1*mean(errs))

	return w_grad, b_grad
}

func mean(x []float64) (r float64) {
	for j := 0; j < len(x); j++ {
		r += x[j]
	}
	return r / float64(len(x))
}

func mulMulti(a *mat64.Dense, b []float64, rows int) (r []float64) {
	var m, m2 mat64.Dense

	b1 := mat64.NewDense(1, 1, []float64{b[0]})
	b2 := mat64.NewDense(1, 1, []float64{b[1]})

	m.Mul(a.ColView(0), b1)
	m2.Mul(a.ColView(1), b2)

	for i := 0; i < rows; i++ {
		r = append(r, m.ColView(0).At(i,0)+m2.ColView(0).At(i,0))
	}
	return r
}
