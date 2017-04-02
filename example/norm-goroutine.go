package main

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"time"

	"../"
	"github.com/gonum/matrix/mat64"
)

func main() {
	cpus := runtime.NumCPU()
	runtime.GOMAXPROCS(cpus)
	fmt.Println("Your machine has", cpus, "cores")

	rand.Seed(int64(0))

	N := 10000
	goroutines := 4
	batch_size := 10

	offset := 5.0
	pts := grad.MakeNormalPoints(N, 0.0)
	pts2 := grad.MakeNormalPoints(N, offset)
	data := mat64.NewDense(N*2, 3, grad.CreateData(pts, pts2))

	// Divide Dataset
	x, y := grad.DivideData(data, 0, N)

	w1 := []float64{}
	w2 := []float64{}
	b := []float64{}

	start := time.Now().UnixNano()

	// Concurrent processing with multiple workers
	receiver := worker(N, goroutines, batch_size, data, x, y)
	for i := 0; i < goroutines; i++ {
		r := <-receiver
		w1 = append(w1, r.Weight[0])
		w2 = append(w2, r.Weight[1])
		b = append(b, r.Intercept)
	}

	// mean of each parameters
	weights := []float64{grad.Mean(w1), grad.Mean(w2)}
	intercept := grad.Mean(b)

	fmt.Println("mean of weights =", weights, "mean of intercept =", intercept)

	end := time.Now().UnixNano()
	fmt.Println(float64(end-start)/float64(1000000), "ms")

	// Plot result
	err := grad.PlotSample(pts, pts2, "../img/msgd-norm-goroutine.png")
	if err != nil {
		panic(err)
	}
	err = grad.PlotLine(pts, pts2, weights, intercept, "../img/msgd-line-goroutine.png")
	if err != nil {
		panic(err)
	}
}

func worker(N, goroutines, batch_size int, data, x *mat64.Dense, y []float64) <-chan *grad.Grad {
	receiver := make(chan *grad.Grad)

	for i := 0; i < goroutines; i++ {
		go func(i int) {
			eta := 0.2
			g := grad.NewGrad(eta)

			offset := i * N / goroutines

			//  Minibatch SGD (minibatch stochastic gradient descent)
			for j := 0; j < N/(goroutines*batch_size); j++ {
				x_part, y_part := grad.DivideData(data, j*batch_size+offset, batch_size)
				w_grad, b_grad := g.Grad(x_part, y_part, batch_size)

				// Update parameters
				g.Weight[0] -= eta * w_grad[0]
				g.Weight[1] -= eta * w_grad[1]
				g.Intercept -= eta * b_grad[0]

				yhat := g.CalcLikeLihood(x, N)
				err_sum := 0.0
				for k := 0; k < len(yhat); k++ {
					err_sum += math.Abs(y[k] - yhat[k])
				}

				g.Errors = append(g.Errors, err_sum/float64(len(yhat)))
			}

			fmt.Println("goroutine =", i, "batch-size =", batch_size)
			fmt.Println("weight =", g.Weight, "b =", g.Intercept)
			fmt.Println("final error =", g.Errors[N/(goroutines*batch_size)-1], "\n")

			receiver <- g
		}(i)
	}

	return receiver
}
