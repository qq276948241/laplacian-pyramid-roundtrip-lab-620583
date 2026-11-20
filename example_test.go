package imaging_test

import (
	"context"
	"fmt"
	"image"
	"image/color"
	"log"

	"github.com/disintegration/imaging"
)

// ExampleBuildGaussianPyramid shows how to build a Gaussian pyramid and inspect
// the size of each level.
func ExampleBuildGaussianPyramid() {
	src, err := imaging.Open("testdata/flowers_small.png")
	if err != nil {
		log.Fatalf("failed to open image: %v", err)
	}

	levels, err := imaging.BuildGaussianPyramid(src, 4)
	if err != nil {
		log.Fatalf("failed to build pyramid: %v", err)
	}
	for i, lv := range levels {
		b := lv.Bounds()
		fmt.Printf("level %d: %dx%d\n", i, b.Dx(), b.Dy())
	}
}

// ExampleDecomposeLaplacian shows a reversible Laplacian decomposition: the
// detail layers can be discarded, stored, and later recombined with the base
// to reproduce the original image exactly.
func ExampleDecomposeLaplacian() {
	src, err := imaging.Open("testdata/flowers_small.png")
	if err != nil {
		log.Fatalf("failed to open image: %v", err)
	}

	dec, err := imaging.DecomposeLaplacian(src, 4)
	if err != nil {
		log.Fatalf("failed to decompose: %v", err)
	}
	fmt.Printf("levels: %d, base: %dx%d\n",
		dec.Levels(), dec.Base.Bounds().Dx(), dec.Base.Bounds().Dy())

	reconstructed, err := dec.Reconstruct()
	if err != nil {
		log.Fatalf("failed to reconstruct: %v", err)
	}

	// The reconstruction is bit-exact with the original 8-bit samples.
	same := imaging.Clone(src).Bounds() == reconstructed.Bounds()
	fmt.Printf("bounds match: %v\n", same)
}

func ExampleDecomposeLaplacian_cancel() {
	src, err := imaging.Open("testdata/flowers_small.png")
	if err != nil {
		log.Fatalf("failed to open image: %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Simulate cancellation before work starts.

	dec, err := imaging.DecomposeLaplacianContext(ctx, src, 4)
	if dec != nil || err != context.Canceled {
		log.Fatal("expected a nil decomposition and context.Canceled")
	}
	fmt.Println("cancelled:", err)
	// Output: cancelled: context canceled
}

func Example() {
	// Open a test image.
	src, err := imaging.Open("testdata/flowers.png")
	if err != nil {
		log.Fatalf("failed to open image: %v", err)
	}

	// Crop the original image to 300x300px size using the center anchor.
	src = imaging.CropAnchor(src, 300, 300, imaging.Center)

	// Resize the cropped image to width = 200px preserving the aspect ratio.
	src = imaging.Resize(src, 200, 0, imaging.Lanczos)

	// Create a blurred version of the image.
	img1 := imaging.Blur(src, 5)

	// Create a grayscale version of the image with higher contrast and sharpness.
	img2 := imaging.Grayscale(src)
	img2 = imaging.AdjustContrast(img2, 20)
	img2 = imaging.Sharpen(img2, 2)

	// Create an inverted version of the image.
	img3 := imaging.Invert(src)

	// Create an embossed version of the image using a convolution filter.
	img4 := imaging.Convolve3x3(
		src,
		[9]float64{
			-1, -1, 0,
			-1, 1, 1,
			0, 1, 1,
		},
		nil,
	)

	// Create a new image and paste the four produced images into it.
	dst := imaging.New(400, 400, color.NRGBA{0, 0, 0, 0})
	dst = imaging.Paste(dst, img1, image.Pt(0, 0))
	dst = imaging.Paste(dst, img2, image.Pt(0, 200))
	dst = imaging.Paste(dst, img3, image.Pt(200, 0))
	dst = imaging.Paste(dst, img4, image.Pt(200, 200))

	// Save the resulting image as JPEG.
	err = imaging.Save(dst, "testdata/out_example.jpg")
	if err != nil {
		log.Fatalf("failed to save image: %v", err)
	}
}
