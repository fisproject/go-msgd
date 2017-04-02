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
	batch_size := 10 // batch_size=1 equals SGD
	eta := 0.2

	g := grad.NewGrad(eta)

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
	x, y := grad.DivideData(data, 0, N)

	start := time.Now().UnixNano()

	//  Minibatch SGD (minibatch stochastic gradient descent)
	for i := 0; i < N/batch_size; i++ {
		x_part, y_part := grad.DivideData(data, i*batch_size, batch_size)

		// batch processing
		w_grad, b_grad := g.Grad(x_part, y_part, batch_size)

		// Update parameters
		g.Weight[0] -= eta * w_grad[0]
		g.Weight[1] -= eta * w_grad[1]
		g.Intercept -= eta * b_grad[0]

		yhat := g.CalcLikeLihood(x, N)
		err_sum := 0.0
		for i := 0; i < len(yhat); i++ {
			err_sum += math.Abs(y[i] - yhat[i])
		}

		g.Errors = append(g.Errors, err_sum/float64(len(yhat)))
	}

	fmt.Println("length-of-data =", N, "batch-size =", batch_size)
	fmt.Println("weight =", g.Weight, "intercept =", g.Intercept)
	fmt.Println("final error =", g.Errors[N/batch_size-1])

	end := time.Now().UnixNano()
	fmt.Println(float64(end-start)/float64(1000000), "ms")

	// Plot result
	err := grad.PlotSample(pts, pts2, "../img/msgd-norm-seq.png")
	if err != nil {
		panic(err)
	}
	err = grad.PlotErrors(g.Errors, "../img/msgd-trace-error-seq.png")
	if err != nil {
		panic(err)
	}
	err = grad.PlotLine(pts, pts2, g.Weight, g.Intercept, "../img/msgd-line-seq.png")
	if err != nil {
		panic(err)
	}
}
