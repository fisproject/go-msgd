package main

import (
	"fmt"
	"math"
	"math/rand"

	"../"
	"github.com/gonum/matrix/mat64"
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

	// Divide Dataset
	x, y := DivideData(data, 0, N/2)

	//  Minibatch SGD (minibatch stochastic gradient descent)
	for i := 0; i < 10; i++ {
		for j := 0; j < N/batch_size; j++ {
			k := j * batch_size

			x_part, y_part := DivideData(data, k, batch_size)

			w_grad, b_grad := grad.Grad(x_part, y_part, w, b, batch_size)

			// Update parameters
			w[0] -= eta * w_grad[0]
			w[1] -= eta * w_grad[1]
			b -= eta * b_grad[0]

			r := grad.P_y_given_x(x, w, b, N/2)

			err_sum := 0.0
			for i := 0; i < len(r); i++ {
				err_sum += math.Abs(y[i] - r[i])
			}

			errors = append(errors, err_sum/float64(len(y)))

			fmt.Println("iter : i =", i, "j =", j)
			fmt.Println("weight =", w, "b =", b)
			fmt.Println("error =", err_sum/float64(len(y)))
		}
	}

	grad.PlotSample(pts, pts2, "../img/sample-norm.png")
	grad.PlotError(errors, "../img/errors.png")
	grad.PlotLine(pts, pts2, w, b, "../img/line.png")
}

func DivideData(data *mat64.Dense, start, end int) (*mat64.Dense, []float64){
	_x := []float64{}
	y := []float64{}

	for i := start; i < (start + end); i++ {
		_x = append(_x, data.ColView(0).At(i,0)) // x1
		_x = append(_x, data.ColView(1).At(i,0)) // x2
		y = append(y, data.ColView(2).At(i,0)) // label
	}
	x := mat64.NewDense(end, 2, _x) // x1 and x2

	return x, y
}
