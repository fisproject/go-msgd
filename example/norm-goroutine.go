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
	// fmt.Println(mat64.Formatted(data))
	//
	// ⎡   0.3355783633826332     0.3075069223376545                      0⎤
	// ⎢    4.644406713877847       3.18779626663706                      1⎥
	// ⎢   1.3714689033418805    -1.5036720020678844                      0⎥
	// ⎢    5.792039934488384      5.151017687843955                      1⎥
	// ...

	// Divide Dataset
	x, y := DivideData(data, 0, N)

	w1 := []float64{}
	w2 := []float64{}
	b := []float64{}

	start := time.Now().UnixNano()

	receiver := worker(N, goroutines, batch_size, data, x, y)
	for i := 0; i < goroutines; i++ {
		r := <-receiver
		w1 = append(w1, r[0])
		w2 = append(w2, r[1])
		b = append(b, r[2])
	}

	// mean of each parameters
	w_mean := []float64{grad.Mean(w1), grad.Mean(w2)}
	b_mean := grad.Mean(b)

	fmt.Println("mean of weights =", w_mean, "mean of b =", b_mean)

	end := time.Now().UnixNano()
	fmt.Println(float64(end-start)/float64(1000000), "ms") // 18968.966233 ms

	grad.PlotSample(pts, pts2, "../img/msgd-norm-goroutine.png")
	grad.PlotLine(pts, pts2, w_mean, b_mean, "../img/msgd-line-goroutine.png")
}

func worker(N, goroutines, batch_size int, data, x *mat64.Dense, y []float64) <-chan []float64 {
	receiver := make(chan []float64)

	for i := 0; i < goroutines; i++ {
		go func(i int) {
			w := []float64{0.0, 0.0}
			b := 0.0
			eta := 0.2
			errors := []float64{}

			offset := i * N/goroutines

			//  Minibatch SGD (minibatch stochastic gradient descent)
			for j := 0; j < N/ (goroutines * batch_size); j++ {
				x_part, y_part := DivideData(data, j*batch_size+offset, batch_size)
				
				// batch processing
				w_grad, b_grad := grad.Grad(x_part, y_part, w, b, batch_size)

				// Update parameters
				w[0] -= eta * w_grad[0]
				w[1] -= eta * w_grad[1]
				b -= eta * b_grad[0]

				yhat := grad.P_y_given_x(x, w, b, N)
				err_sum := 0.0
				for k := 0; k < len(yhat); k++ {
					err_sum += math.Abs(y[k] - yhat[k])
				}

				errors = append(errors, err_sum/float64(len(yhat)))
			}

			fmt.Println("goroutine =", i, "batch-size =", batch_size)
			fmt.Println("weight =", w, "b =", b)
			fmt.Println("final error =", errors[ N/ (goroutines * batch_size)-1])

			res := []float64{w[0], w[1], b}
			receiver <- res
		}(i)
	}

	return receiver
}

func DivideData(data *mat64.Dense, start, size int) (*mat64.Dense, []float64) {
	_x := []float64{}
	y := []float64{}

	for i := start; i < (start + size); i++ {
		_x = append(_x, data.ColView(0).At(i, 0)) // x1
		_x = append(_x, data.ColView(1).At(i, 0)) // x2
		y = append(y, data.ColView(2).At(i, 0))   // label
	}
	x := mat64.NewDense(size, 2, _x) // x1 and x2

	return x, y
}
