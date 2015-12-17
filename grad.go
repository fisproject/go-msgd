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

func mulEx(a *mat64.Dense, b []float64, rows int) (r []float64) {
	var m, m2 mat64.Dense

	a_t := mat64.DenseCopyOf(a.T())
	a_t_r0 := mat64.NewDense(rows, 1, a_t.RawRowView(0))
	a_t_r1 := mat64.NewDense(rows, 1, a_t.RawRowView(1))

	b0 := mat64.NewDense(1, 1, []float64{b[0]})
	b1 := mat64.NewDense(1, 1, []float64{b[1]})

	m.Mul(a_t_r0, b0)
	m2.Mul(a_t_r1, b1)

	m_t := mat64.DenseCopyOf(m.T())
	m2_t := mat64.DenseCopyOf(m2.T())

	r1 := m_t.RawRowView(0)
	r2 := m2_t.RawRowView(0)

	for i := 0; i < rows; i++ {
		r = append(r, r1[i]+r2[i])
	}
	return r
}

// objective function (likelihood) : p(y|x) = 1 / 1 + exp(-y w x)
func P_y_given_x(x *mat64.Dense, w []float64, b float64, s int) (l []float64) {
	m := mulEx(x, w, s)

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
		errs = append(errs, c[i]-l[i]) // error = label - likelihood
	}
	err_mat := mat64.NewDense(1, s, errs)

	x_t := mat64.DenseCopyOf(x.T())
	x_t_r0 := mat64.NewDense(1, s, x_t.RawRowView(0))
	x_t_r1 := mat64.NewDense(1, s, x_t.RawRowView(1))

	w_grad = append(w_grad, -1*mat64.Dot(x_t_r0, err_mat))
	w_grad = append(w_grad, -1*mat64.Dot(x_t_r1, err_mat))
	b_grad = append(b_grad, -1*mean(errs))

	return w_grad, b_grad
}

func vecToFloat64(v *mat64.Vector) (r []float64) {
	for i := 0; i < len(v.RawVector().Data); {
		r = append(r, v.RawVector().Data[i])
		i += v.RawVector().Inc
	}
	return r
}

func mean(x []float64) (r float64) {
	for j := 0; j < len(x); j++ {
		r += x[j]
	}
	return r / float64(len(x))
}
