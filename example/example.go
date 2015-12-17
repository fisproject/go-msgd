package main

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/gonum/matrix/mat64"
	"../"
)

func main() {
	rand.Seed(int64(0))

	N := 200
	batch_size := 10

	w := []float64{grad.RandNormal(0, 1), grad.RandNormal(0, 1)}
	b := grad.RandNormal(0, 1)
	eta := 0.2
	errors := []float64{}

	offset := 5.0
	pts := grad.MakeNormalPoints(N/2, 0.0)
	pts2 := grad.MakeNormalPoints(N/2, offset)
	data := mat64.NewDense(N, 3, grad.CreateData(pts, pts2))
	// fmt.Println(mat64.Formatted(data))
	//
	// ⎡   0.3355783633826332     0.3075069223376545                      0⎤
	// ⎢    4.644406713877847       3.18779626663706                      1⎥
	// ⎢   1.3714689033418805    -1.5036720020678844                      0⎥
	// ⎢    5.792039934488384      5.151017687843955                      1⎥
	// ...

	data_t := mat64.DenseCopyOf(data.T())
	x_a := data_t.RawRowView(0)[0 : N/2]
	x_a = append(x_a, data_t.RawRowView(1)[0:N/2]...)
	x_a_t := mat64.NewDense(2, N/2, x_a)
	_x_a := mat64.DenseCopyOf(x_a_t.T())
	_c_a := data_t.RawRowView(2)[0 : N/2]

	//  Minibatch SGD (minibatch stochastic gradient descent)
	for i := 0; i < 10; i++ {
		for j := 0; j < N/batch_size; j++ {
			k := j * batch_size

			x := data_t.RawRowView(0)[k : k+batch_size]
			x = append(x, data_t.RawRowView(1)[k:k+batch_size]...)

			_x_t := mat64.NewDense(2, batch_size, x)
			_x := mat64.DenseCopyOf(_x_t.T())
			_c := data_t.RawRowView(2)[k : k+batch_size]

			w_grad, b_grad := grad.Grad(_x, _c, w, b, batch_size)

			// Update parameters
			w[0] -= eta * w_grad[0]
			w[1] -= eta * w_grad[1]
			b -= eta * b_grad[0]

			err_sum := 0.0
			r := grad.P_y_given_x(_x_a, w, b, N/2)
			for i := 0; i < len(r); i++ {
				err := _c_a[i] - r[i]
				err_sum += math.Abs(err)
			}
			errors = append(errors, err_sum/float64(len(_c_a)))

			fmt.Println("iter : i =", i, "j =", j)
			fmt.Println("weight =", w, "b =", b)
			fmt.Println("error =", err_sum/float64(len(_c_a)))
		}
	}

	grad.PlotSample(pts, pts2, "../img/sample-norm.png")
	grad.PlotError(errors, "../img/errors.png")
	grad.PlotLine(pts, pts2, w, b, "../img/line.png")
}
