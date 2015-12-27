package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"../"
	"github.com/gonum/matrix/mat64"
)

func main() {
	rand.Seed(int64(0))

	N := 10000
	batch_size := 10

	w := []float64{grad.RandNormal(0, 1), grad.RandNormal(0, 1)}
	b := grad.RandNormal(0, 1)
	eta := 0.2
	errors := []float64{}

	offset := 5.0
	pts := grad.MakeNormalPoints(N, 0.0)
	pts2 := grad.MakeNormalPoints(N, offset)
	data := mat64.NewDense(N*2, 3, grad.CreateData(pts, pts2))
	// fmt.Println(mat64.Formatted(data))
	//
	// ⎡   0.3355783633826332     0.3075069223376545                      0⎤
	// ⎢    4.644406713877847       3.18779626663706                      1⎥
	// ⎢   1.3714689033418805    -1.5036720020678844                      0⎥
	// ⎢    5.792039934488384      5.151017687843955                      1⎥
	// ...

	// Divide Dataset
	x, y := DivideData(data, 0, N)

	start := time.Now().UnixNano()

	//  Minibatch SGD (minibatch stochastic gradient descent)
	for i := 0; i < batch_size; i++ {
		k := N / batch_size
		for j := i * k; j < i*k+k; j++ {

			x_part, y_part := DivideData(data, j, batch_size)

			w_grad, b_grad := grad.Grad(x_part, y_part, w, b, batch_size)

			// Update parameters
			w[0] -= eta * w_grad[0]
			w[1] -= eta * w_grad[1]
			b -= eta * b_grad[0]

			r := grad.P_y_given_x(x, w, b, N)

			err_sum := 0.0
			for i := 0; i < len(r); i++ {
				err_sum += math.Abs(y[i] - r[i])
			}

			errors = append(errors, err_sum/float64(len(y)))
		}
		fmt.Println("iter =", i, "size =", k)
		fmt.Println("weight =", w, "b =", b)
		fmt.Println("final error =", errors[((i+1)*k)-1])
	}

	end := time.Now().UnixNano()
	fmt.Println(float64(end-start)/float64(1000000), "ms") // 547.321513 ms

	grad.PlotSample(pts, pts2, "../img/msgd-norm-seq.png")
	grad.PlotError(errors, "../img/msgd-trace-error-seq.png")
	grad.PlotLine(pts, pts2, w, b, "../img/msgd-line-seq.png")
}

func DivideData(data *mat64.Dense, start, end int) (*mat64.Dense, []float64) {
	_x := []float64{}
	y := []float64{}

	for i := start; i < (start + end); i++ {
		_x = append(_x, data.ColView(0).At(i, 0)) // x1
		_x = append(_x, data.ColView(1).At(i, 0)) // x2
		y = append(y, data.ColView(2).At(i, 0))   // label
	}
	x := mat64.NewDense(end, 2, _x) // x1 and x2

	return x, y
}
