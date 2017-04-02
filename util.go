package grad

import (
	"math"
	"math/rand"

	"github.com/gonum/matrix/mat64"
	"github.com/gonum/plot"
	"github.com/gonum/plot/plotter"
	"github.com/gonum/plot/plotutil"
	"github.com/gonum/plot/vg"
)

// TODO : need shuffle
func CreateData(d1, d2 plotter.XYs) (d []float64) {
	for i := range d1 {
		d = append(d, d1[i].X)
		d = append(d, d1[i].Y)
		d = append(d, 0.0)

		d = append(d, d2[i].X)
		d = append(d, d2[i].Y)
		d = append(d, 1.0)
	}
	return d
}

func DivideData(data *mat64.Dense, start, size int) (x *mat64.Dense, y []float64) {
	tmp := []float64{}

	for i := start; i < (start + size); i++ {
		tmp = append(tmp, data.ColView(0).At(i, 0)) // x1
		tmp = append(tmp, data.ColView(1).At(i, 0)) // x2
		y = append(y, data.ColView(2).At(i, 0))     // label
	}
	x = mat64.NewDense(size, 2, tmp) // x1 and x2

	return x, y
}

func Mean(x []float64) (y float64) {
	sum := 0.0
	for i := 0; i < len(x); i++ {
		sum += x[i]
	}
	return sum / float64(len(x))
}

// Loss function
func Sigmoid(x float64) (y float64) {
	return 1.0 / (1.0 + math.Exp(-1*x))
}

func UnSigmoid(x float64) (y float64) {
	x = x*0.99 + 0.01
	return math.Log(x / (1 - x))
}

func MulMulti(a *mat64.Dense, b []float64, rows int) (ret []float64) {
	var m, m2 mat64.Dense

	b1 := mat64.NewDense(1, 1, []float64{b[0]})
	b2 := mat64.NewDense(1, 1, []float64{b[1]})

	m.Mul(a.ColView(0), b1)
	m2.Mul(a.ColView(1), b2)

	for i := 0; i < rows; i++ {
		ret = append(ret, m.ColView(0).At(i, 0)+m2.ColView(0).At(i, 0))
	}
	return ret
}

func RandNormal(mu, sigma float64) float64 {
	z := math.Sqrt(-2.0*math.Log(rand.Float64())) * math.Sin(2.0*math.Pi*rand.Float64())
	return sigma*z + mu
}

func MakeNormalPoints(n int, offset float64) plotter.XYs {
	pts := make(plotter.XYs, n)
	for i := range pts {
		pts[i].X = RandNormal(0, 1) + offset
		pts[i].Y = RandNormal(0, 1) + offset
	}
	return pts
}

func makePoints(y []float64) plotter.XYs {
	pts := make(plotter.XYs, len(y))
	for i := range pts {
		pts[i].X = float64(i)
		pts[i].Y = y[i]
	}
	return pts
}

func makeLine(x, y []float64) plotter.XYs {
	pts := make(plotter.XYs, len(x))
	for i := range pts {
		pts[i].X = x[i]
		pts[i].Y = y[i]
	}
	return pts
}

func PlotSample(pts, pts2 plotter.XYs, path string) error {
	p, err := plot.New()
	if err != nil {
		return err
	}

	p.X.Label.Text = "x"
	p.Y.Label.Text = "y"

	err = plotutil.AddScatters(p, "Class 0", pts, "Class 1", pts2)
	if err != nil {
		return err
	}

	return p.Save(10*vg.Inch, 10*vg.Inch, path)
}

func PlotErrors(errors []float64, path string) error {
	p, err := plot.New()
	if err != nil {
		return err
	}

	p.X.Label.Text = "iteration"
	p.Y.Label.Text = "error"

	pts_err := makePoints(errors)
	err = plotutil.AddScatters(p, "Error", pts_err)
	if err != nil {
		return err
	}

	return p.Save(10*vg.Inch, 10*vg.Inch, path)
}

func PlotLine(pts, pts2 plotter.XYs, w []float64, b float64, path string) error {
	var x, y []float64

	p, err := plot.New()
	if err != nil {
		return err
	}

	p.X.Label.Text = "x"
	p.Y.Label.Text = "y"
	for i := range make([]float64, 150) {
		bx := float64(i)*0.1 - 6.0
		x = append(x, bx)
		y = append(y, -b/w[1]-w[0]/w[1]*bx)
	}

	pts_res := makeLine(x, y)
	err = plotutil.AddScatters(p, "Class 0", pts, "Class 1", pts2)
	if err != nil {
		return err
	}

	err = plotutil.AddLines(p, "Line", pts_res)
	if err != nil {
		return err
	}

	return p.Save(10*vg.Inch, 10*vg.Inch, path)
}
