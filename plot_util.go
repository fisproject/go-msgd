package main

import (
	"math"
	"math/rand"

	"github.com/gonum/plot"
	"github.com/gonum/plot/plotter"
	"github.com/gonum/plot/plotutil"
	"github.com/gonum/plot/vg"
)

// TODO : need shuffle
func createData(d1, d2 plotter.XYs) (d []float64) {
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

func randNormal(mu, sigma float64) float64 {
	z := math.Sqrt(-2.0*math.Log(rand.Float64())) * math.Sin(2.0*math.Pi*rand.Float64())
	return sigma*z + mu
}

func makeNormalPoints(n int, offset float64) plotter.XYs {
	pts := make(plotter.XYs, n)
	for i := range pts {
		pts[i].X = randNormal(0, 1) + offset
		pts[i].Y = randNormal(0, 1) + offset
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

func plotSample(pts, pts2 plotter.XYs) {
	p, err := plot.New()
	if err != nil {
		panic(err)
	}
	p.X.Label.Text = "x"
	p.Y.Label.Text = "y"
	err = plotutil.AddScatters(p,
		"Class 0", pts,
		"Class 1", pts2,
	)
	if err != nil {
		panic(err)
	}
	if err := p.Save(4*vg.Inch, 4*vg.Inch, "img/sample-norm.png"); err != nil {
		panic(err)
	}
}

func plotError(errors []float64) {
	p, err := plot.New()
	if err != nil {
		panic(err)
	}
	p.X.Label.Text = "iteration"
	p.Y.Label.Text = "error"
	pts_err := makePoints(errors)
	err = plotutil.AddScatters(p,
		"Error", pts_err,
	)
	if err != nil {
		panic(err)
	}
	if err := p.Save(4*vg.Inch, 4*vg.Inch, "img/errors.png"); err != nil {
		panic(err)
	}
}

func plotLine(pts, pts2 plotter.XYs, w []float64, b float64) {
	p, err := plot.New()
	if err != nil {
		panic(err)
	}
	p.X.Label.Text = "x"
	p.Y.Label.Text = "y"

	x := []float64{}
	y := []float64{}
	for i := range make([]float64, 150) {
		bx := float64(i)*0.1 - 6.0
		x = append(x, bx)
		y = append(y, -b/w[1]-w[0]/w[1]*bx)
	}
	pts_res := makeLine(x, y)
	err = plotutil.AddScatters(p,
		"Class 0", pts,
		"Class 1", pts2,
	)
	if err != nil {
		panic(err)
	}
	err = plotutil.AddLines(p,
		"Line", pts_res,
	)
	if err != nil {
		panic(err)
	}
	if err := p.Save(4*vg.Inch, 4*vg.Inch, "img/line.png"); err != nil {
		panic(err)
	}
}
