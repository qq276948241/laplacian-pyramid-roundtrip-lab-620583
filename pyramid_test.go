package imaging

import (
	"context"
	"errors"
	"image"
	"image/color"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

// mustNRGBA builds a compact NRGBA image with the given dimensions.
func mustNRGBA(w, h int) *image.NRGBA {
	return image.NewNRGBA(image.Rect(0, 0, w, h))
}

// setPix writes an NRGBA sample at (x, y) of img, using img's own stride.
func setPix(img *image.NRGBA, x, y int, c color.NRGBA) {
	o := (y-img.Rect.Min.Y)*img.Stride + (x-img.Rect.Min.X)*4
	img.Pix[o+0] = c.R
	img.Pix[o+1] = c.G
	img.Pix[o+2] = c.B
	img.Pix[o+3] = c.A
}

func getPix(img *image.NRGBA, x, y int) color.NRGBA {
	o := (y-img.Rect.Min.Y)*img.Stride + (x-img.Rect.Min.X)*4
	return color.NRGBA{img.Pix[o], img.Pix[o+1], img.Pix[o+2], img.Pix[o+3]}
}

// pixelsEqual reports the first byte difference between two NRGBA images after
// comparing their bounds. An empty string means the images are byte-identical.
func pixelsEqual(a, b *image.NRGBA) string {
	if a.Bounds() != b.Bounds() {
		return "bounds differ"
	}
	if a.Stride != b.Stride {
		return "stride differs"
	}
	for i := range a.Pix {
		if a.Pix[i] != b.Pix[i] {
			x := (i % a.Stride) / 4
			y := i / a.Stride
			return "pixel differs at (" + itoa(x) + "," + itoa(y) + ")"
		}
	}
	return ""
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	neg := n < 0
	if neg {
		n = -n
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		buf[i] = '-'
	}
	return string(buf[i:])
}

// gradientNRGBA creates a deterministic, non-constant image with varied alpha.
func gradientNRGBA(w, h int) *image.NRGBA {
	img := mustNRGBA(w, h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			setPix(img, x, y, color.NRGBA{
				R: uint8((x*37 + y*11) & 0xff),
				G: uint8((x*13 + y*53) & 0xff),
				B: uint8((x*97 + y*7) & 0xff),
				A: uint8((x*29 + y*31 + 17) & 0xff),
			})
		}
	}
	return img
}

// paddedSubImage builds a larger image, returns a SubImage with a non-zero
// origin, and verifies the padded strides differ from a compact image.
func paddedSubImage(w, h int) (*image.NRGBA, *image.NRGBA) {
	big := gradientNRGBA(w+3, h+5)
	sub := big.SubImage(image.Rect(2, 3, 2+w, 3+h)).(*image.NRGBA)
	return big, sub
}

// sameContent reports the first difference between two NRGBA images comparing
// size and per-pixel values relative to each image's own origin. Origins may
// differ; pixel content may not.
func sameContent(a, b *image.NRGBA) string {
	if a.Bounds().Dx() != b.Bounds().Dx() || a.Bounds().Dy() != b.Bounds().Dy() {
		return "size differs"
	}
	for y := 0; y < a.Bounds().Dy(); y++ {
		for x := 0; x < a.Bounds().Dx(); x++ {
			if getPix(a, x+a.Bounds().Min.X, y+a.Bounds().Min.Y) != getPix(b, x+b.Bounds().Min.X, y+b.Bounds().Min.Y) {
				return "pixel differs at (" + itoa(x) + "," + itoa(y) + ")"
			}
		}
	}
	return ""
}

// cloneBytes snapshots the pixel storage of an image so tests can prove the
// source was not written to.
func cloneBytes(img *image.NRGBA) []byte {
	return append([]byte(nil), img.Pix...)
}

// -----------------------------------------------------------------------------
// PyramidSizes: size rules and errors
// -----------------------------------------------------------------------------

func TestPyramidSizes(t *testing.T) {
	cases := []struct {
		w, h   int
		levels int
		want   []image.Point
	}{
		{1, 1, 1, []image.Point{{1, 1}}},
		{2, 2, 2, []image.Point{{2, 2}, {1, 1}}},
		{3, 3, 3, []image.Point{{3, 3}, {2, 2}, {1, 1}}},
		// Odd rows/columns are rounded up, never truncated.
		{5, 3, 4, []image.Point{{5, 3}, {3, 2}, {2, 1}, {1, 1}}},
		{1, 8, 4, []image.Point{{1, 8}, {1, 4}, {1, 2}, {1, 1}}},
		{7, 1, 3, []image.Point{{7, 1}, {4, 1}, {2, 1}}},
		{4, 4, 3, []image.Point{{4, 4}, {2, 2}, {1, 1}}},
	}
	for _, c := range cases {
		got, err := PyramidSizes(c.w, c.h, c.levels)
		if err != nil {
			t.Fatalf("PyramidSizes(%d,%d,%d): %v", c.w, c.h, c.levels, err)
		}
		if len(got) != len(c.want) {
			t.Fatalf("PyramidSizes(%d,%d,%d) length = %d, want %d", c.w, c.h, c.levels, len(got), len(c.want))
		}
		for i := range got {
			if got[i] != c.want[i] {
				t.Fatalf("PyramidSizes(%d,%d,%d)[%d] = %v, want %v", c.w, c.h, c.levels, i, got[i], c.want[i])
			}
		}
	}
}

func TestPyramidSizesErrors(t *testing.T) {
	cases := []struct {
		name         string
		w, h, levels int
		want         error
	}{
		{"zero width", 0, 10, 1, ErrEmptyImage},
		{"negative height", 10, -1, 1, ErrEmptyImage},
		{"zero levels", 10, 10, 0, ErrInvalidLevels},
		{"negative levels", 10, 10, -3, ErrInvalidLevels},
		{"too many levels 4x4", 4, 4, 4, ErrTooManyLevels},
		{"too many levels 1x1", 1, 1, 2, ErrTooManyLevels},
		{"too many levels 1x8", 1, 8, 5, ErrTooManyLevels},
	}
	for _, c := range cases {
		_, err := PyramidSizes(c.w, c.h, c.levels)
		if !errors.Is(err, c.want) {
			t.Errorf("%s: got %v want %v", c.name, err, c.want)
		}
	}
}

// TestMaxPyramidLevelsExact checks that the largest accepted level count is
// exactly the number of valid downsample steps.
func TestMaxPyramidLevelsExact(t *testing.T) {
	for w := 1; w <= 33; w++ {
		for h := 1; h <= 33; h++ {
			max := maxPyramidLevels(w, h)
			if _, err := PyramidSizes(w, h, max); err != nil {
				t.Fatalf("%dx%d: max levels %d rejected: %v", w, h, max, err)
			}
			if _, err := PyramidSizes(w, h, max+1); !errors.Is(err, ErrTooManyLevels) {
				t.Fatalf("%dx%d: levels %d should be too many, got %v", w, h, max+1, err)
			}
		}
	}
}

// -----------------------------------------------------------------------------
// Error contract: nil/empty images and budget before allocation
// -----------------------------------------------------------------------------

type fakeImage struct {
	r image.Rectangle
}

func (f fakeImage) ColorModel() color.Model { return color.RGBAModel }
func (f fakeImage) Bounds() image.Rectangle { return f.r }
func (f fakeImage) At(x, y int) color.Color {
	panic("At must not be called on a budget-rejected image")
}

func TestDecompositionErrors(t *testing.T) {
	if _, err := BuildGaussianPyramid(nil, 1); !errors.Is(err, ErrNilImage) {
		t.Errorf("nil image: got %v want %v", err, ErrNilImage)
	}
	if _, err := DecomposeLaplacian(nil, 1); !errors.Is(err, ErrNilImage) {
		t.Errorf("nil image: got %v want %v", err, ErrNilImage)
	}

	empty := &image.NRGBA{}
	if _, err := BuildGaussianPyramid(empty, 1); !errors.Is(err, ErrEmptyImage) {
		t.Errorf("empty image: got %v want %v", err, ErrEmptyImage)
	}
	if _, err := DecomposeLaplacian(empty, 1); !errors.Is(err, ErrEmptyImage) {
		t.Errorf("empty image: got %v want %v", err, ErrEmptyImage)
	}

	img := mustNRGBA(8, 8)
	if _, err := BuildGaussianPyramid(img, 0); !errors.Is(err, ErrInvalidLevels) {
		t.Errorf("zero levels: got %v want %v", err, ErrInvalidLevels)
	}
	if _, err := DecomposeLaplacian(img, 0); !errors.Is(err, ErrInvalidLevels) {
		t.Errorf("zero levels: got %v want %v", err, ErrInvalidLevels)
	}
	if _, err := BuildGaussianPyramid(img, 100); !errors.Is(err, ErrTooManyLevels) {
		t.Errorf("too many levels: got %v want %v", err, ErrTooManyLevels)
	}
	if _, err := DecomposeLaplacian(img, 100); !errors.Is(err, ErrTooManyLevels) {
		t.Errorf("too many levels: got %v want %v", err, ErrTooManyLevels)
	}
}

// TestBudgetRejectedBeforeAllocation ensures the budget check happens before
// any image data is touched: the fake image panics if At is ever called.
func TestBudgetRejectedBeforeAllocation(t *testing.T) {
	// 20000*20000*4 bytes = 1.6 GB, above the 1 GiB budget.
	huge := fakeImage{image.Rect(0, 0, 20000, 20000)}
	if _, err := BuildGaussianPyramid(huge, 1); !errors.Is(err, ErrImageTooLarge) {
		t.Errorf("gaussian: got %v want %v", err, ErrImageTooLarge)
	}
	if _, err := DecomposeLaplacian(huge, 1); !errors.Is(err, ErrImageTooLarge) {
		t.Errorf("laplacian: got %v want %v", err, ErrImageTooLarge)
	}

	// Dimensions that would overflow a naive w*h computation must also be
	// rejected rather than wrapping around.
	maxInt := int(^uint(0) >> 1)
	overflow := fakeImage{image.Rect(0, 0, maxInt, maxInt)}
	if _, err := BuildGaussianPyramid(overflow, 1); !errors.Is(err, ErrImageTooLarge) {
		t.Errorf("overflow: got %v want %v", err, ErrImageTooLarge)
	}
	// Direct budget helpers must be overflow-safe too.
	if err := checkGaussianBudget([]image.Point{{maxInt, maxInt}}); !errors.Is(err, ErrImageTooLarge) {
		t.Errorf("checkGaussianBudget overflow: got %v want %v", err, ErrImageTooLarge)
	}
	if err := checkLaplacianBudget([]image.Point{{maxInt, maxInt}}); !errors.Is(err, ErrImageTooLarge) {
		t.Errorf("checkLaplacianBudget overflow: got %v want %v", err, ErrImageTooLarge)
	}
}

// counterCtx cancels deterministically after a fixed number of Err calls.
type counterCtx struct {
	calls int64
	after int64
}

func (c *counterCtx) Deadline() (deadline time.Time, ok bool) { return time.Time{}, false }
func (c *counterCtx) Done() <-chan struct{}                   { return nil }
func (c *counterCtx) Value(key interface{}) interface{}       { return nil }
func (c *counterCtx) Err() error {
	if atomic.AddInt64(&c.calls, 1) > c.after {
		return context.Canceled
	}
	return nil
}

func TestCancellationReturnsNoPartialResult(t *testing.T) {
	img := gradientNRGBA(64, 48)

	// Already-canceled context.
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if levels, err := BuildGaussianPyramidContext(canceled, img, 4); levels != nil || !errors.Is(err, context.Canceled) {
		t.Errorf("gaussian: got levels=%v err=%v want nil/context.Canceled", levels, err)
	}
	dec, err := DecomposeLaplacianContext(canceled, img, 4)
	if dec != nil || !errors.Is(err, context.Canceled) {
		t.Errorf("laplacian: got dec=%v err=%v want nil/context.Canceled", dec, err)
	}

	// Cancellation partway through: some work has been done but no partial
	// pyramid may escape.
	for _, after := range []int64{0, 1, 2, 3} {
		ctx := &counterCtx{after: after}
		levels, err := BuildGaussianPyramidContext(ctx, img, 5)
		if levels != nil {
			t.Fatalf("after=%d: gaussian returned partial levels", after)
		}
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("after=%d: gaussian err = %v, want context.Canceled", after, err)
		}
		ctx = &counterCtx{after: after}
		dec, err := DecomposeLaplacianContext(ctx, img, 5)
		if dec != nil {
			t.Fatalf("after=%d: laplacian returned partial decomposition", after)
		}
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("after=%d: laplacian err = %v, want context.Canceled", after, err)
		}
	}
}

// -----------------------------------------------------------------------------
// Gaussian pyramid: pixel-level behavior
// -----------------------------------------------------------------------------

func TestGaussianPyramidSizesAndBounds(t *testing.T) {
	src := gradientNRGBA(9, 6)
	levels, err := BuildGaussianPyramid(src, 4)
	if err != nil {
		t.Fatal(err)
	}
	want := []image.Point{{9, 6}, {5, 3}, {3, 2}, {2, 1}}
	if len(levels) != len(want) {
		t.Fatalf("got %d levels want %d", len(levels), len(want))
	}
	for i, lv := range levels {
		if lv.Bounds() != image.Rect(0, 0, want[i].X, want[i].Y) {
			t.Fatalf("level %d bounds = %v want %v", i, lv.Bounds(), image.Rect(0, 0, want[i].X, want[i].Y))
		}
	}
}

func TestGaussianPyramidConstantInvariant(t *testing.T) {
	constant := color.NRGBA{R: 100, G: 150, B: 200, A: 128}
	src := mustNRGBA(11, 7)
	for y := 0; y < 7; y++ {
		for x := 0; x < 11; x++ {
			setPix(src, x, y, constant)
		}
	}
	levels, err := BuildGaussianPyramid(src, 4)
	if err != nil {
		t.Fatal(err)
	}
	for i, lv := range levels {
		for y := lv.Bounds().Min.Y; y < lv.Bounds().Max.Y; y++ {
			for x := lv.Bounds().Min.X; x < lv.Bounds().Max.X; x++ {
				if got := getPix(lv, x, y); got != constant {
					t.Fatalf("level %d pixel (%d,%d) = %v want %v", i, x, y, got, constant)
				}
			}
		}
	}
}

// TestGaussianPyramidAlphaNoLeak verifies that RGB stored under fully
// transparent pixels does not bleed into visible pixels.
func TestGaussianPyramidAlphaNoLeak(t *testing.T) {
	src := mustNRGBA(5, 1)
	// Hidden pure red under alpha 0 next to opaque white.
	setPix(src, 0, 0, color.NRGBA{R: 255, G: 0, B: 0, A: 0})
	setPix(src, 1, 0, color.NRGBA{R: 255, G: 0, B: 0, A: 0})
	setPix(src, 2, 0, color.NRGBA{R: 255, G: 255, B: 255, A: 255})
	setPix(src, 3, 0, color.NRGBA{R: 0, G: 255, B: 0, A: 0})
	setPix(src, 4, 0, color.NRGBA{R: 0, G: 0, B: 255, A: 0})

	levels, err := BuildGaussianPyramid(src, 2)
	if err != nil {
		t.Fatal(err)
	}
	lv := levels[1]
	for x := lv.Bounds().Min.X; x < lv.Bounds().Max.X; x++ {
		p := getPix(lv, x, 0)
		if p.A == 0 {
			continue
		}
		if p.R != p.G || p.G != p.B {
			t.Fatalf("visible pixel (%d) picked up hidden chroma: %v", x, p)
		}
	}
}

// TestGaussianPyramidImpulse checks a single bright pixel spreads symmetrically
// and keeps total energy bounded by the normalized kernel.
func TestGaussianPyramidImpulse(t *testing.T) {
	src := mustNRGBA(5, 5)
	setPix(src, 2, 2, color.NRGBA{R: 255, G: 255, B: 255, A: 255})
	blurred, err := gaussianBlur3x3Context(context.Background(), src)
	if err != nil {
		t.Fatal(err)
	}
	// Filtering is premultiplied, so the non-premultiplied RGB of a lone white
	// impulse stays 255 while alpha spreads by the kernel weight.
	// Center: alpha 4/16*255 = 63.75 -> round half up -> 64.
	if p := getPix(blurred, 2, 2); p != (color.NRGBA{255, 255, 255, 64}) {
		t.Fatalf("center = %v want {255,255,255,64}", p)
	}
	// Edge tap: alpha 2/16*255 = 31.875 -> 32.
	if p := getPix(blurred, 1, 2); p != (color.NRGBA{255, 255, 255, 32}) {
		t.Fatalf("neighbor = %v want {255,255,255,32}", p)
	}
	// Symmetry across both axes.
	for dy := -2; dy <= 2; dy++ {
		for dx := -2; dx <= 2; dx++ {
			if getPix(blurred, 2+dx, 2+dy) != getPix(blurred, 2-dx, 2-dy) {
				t.Fatalf("impulse response not symmetric at (%d,%d)", dx, dy)
			}
		}
	}
}

// TestGaussianPyramidSinglePixelAndRow covers degenerate dimensions.
func TestGaussianPyramidSinglePixelAndRow(t *testing.T) {
	for _, tc := range []struct{ w, h int }{{1, 1}, {1, 9}, {9, 1}, {2, 2}} {
		src := gradientNRGBA(tc.w, tc.h)
		max := maxPyramidLevels(tc.w, tc.h)
		levels, err := BuildGaussianPyramid(src, max)
		if err != nil {
			t.Fatalf("%dx%d: %v", tc.w, tc.h, err)
		}
		last := levels[len(levels)-1]
		if last.Bounds() != image.Rect(0, 0, 1, 1) {
			t.Fatalf("%dx%d: final level bounds = %v want 1x1", tc.w, tc.h, last.Bounds())
		}
	}
}

// TestGaussianPyramidSourceUnchanged proves the source pixel storage is not
// modified, including for SubImages.
func TestGaussianPyramidSourceUnchanged(t *testing.T) {
	big, sub := paddedSubImage(9, 7)
	before := cloneBytes(big)
	if _, err := BuildGaussianPyramid(sub, 3); err != nil {
		t.Fatal(err)
	}
	for i := range before {
		if big.Pix[i] != before[i] {
			t.Fatalf("source modified at byte %d", i)
		}
	}
}

// TestGaussianPyramidSubImageAndStride asserts non-zero origins and padded
// strides produce exactly the same levels as a compact clone.
func TestGaussianPyramidSubImageAndStride(t *testing.T) {
	_, sub := paddedSubImage(9, 7)
	if sub.Bounds().Min == (image.Point{}) {
		t.Fatal("test setup: expected non-zero origin")
	}
	got, err := BuildGaussianPyramid(sub, 4)
	if err != nil {
		t.Fatal(err)
	}
	want, err := BuildGaussianPyramid(Clone(sub), 4)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(want) {
		t.Fatalf("level count %d want %d", len(got), len(want))
	}
	for i := range got {
		if got[i].Bounds() != image.Rect(0, 0, got[i].Bounds().Dx(), got[i].Bounds().Dy()) {
			t.Fatalf("level %d not origin-normalized: %v", i, got[i].Bounds())
		}
		if diff := pixelsEqual(got[i], want[i]); diff != "" {
			t.Fatalf("level %d: %s", i, diff)
		}
	}
}

// TestGaussianPyramidColorModels checks that the decomposition operates on the
// same 8-bit channel semantics as Clone/newScanner for other color models.
func TestGaussianPyramidColorModels(t *testing.T) {
	base := gradientNRGBA(8, 6)

	rgba := image.NewRGBA(base.Bounds())
	for y := 0; y < 6; y++ {
		for x := 0; x < 8; x++ {
			// Premultiplied conversion from NRGBA.
			c := getPix(base, x, y)
			rgba.Set(x, y, c)
		}
	}
	gray := image.NewGray(base.Bounds())
	for y := 0; y < 6; y++ {
		for x := 0; x < 8; x++ {
			gray.SetGray(x, y, color.Gray{Y: uint8(x*17 + y*23)})
		}
	}
	gray16 := image.NewGray16(base.Bounds())
	for y := 0; y < 6; y++ {
		for x := 0; x < 8; x++ {
			gray16.SetGray16(x, y, color.Gray16{Y: uint16(x*4096 + y*2048 + 123)})
		}
	}
	env := image.NewYCbCr(base.Bounds(), image.YCbCrSubsampleRatio444)
	for y := 0; y < 6; y++ {
		for x := 0; x < 8; x++ {
			env.Y[env.YOffset(x, y)] = uint8(x*20 + y*10)
			env.Cb[env.COffset(x, y)] = uint8(128 + x)
			env.Cr[env.COffset(x, y)] = uint8(128 - y)
		}
	}

	sources := map[string]image.Image{
		"RGBA":   rgba,
		"Gray":   gray,
		"Gray16": gray16,
		"YCbCr":  env,
	}
	for name, src := range sources {
		levels, err := BuildGaussianPyramid(src, 3)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		// Level 0 must match the package's existing conversion semantics.
		want := Clone(src)
		if diff := pixelsEqual(levels[0], want); diff != "" {
			t.Fatalf("%s: level 0: %s", name, diff)
		}
	}
}

// -----------------------------------------------------------------------------
// Naive reference implementation
// -----------------------------------------------------------------------------

// naiveGaussianBlur is a straightforward per-pixel implementation of the fixed
// 3x3 Gaussian analysis kernel. It accumulates rgba values premultiplied by
// alpha, samples outside the image via replication, and rounds with
// floor(x+0.5). Taps are visited in row-major order to match the documented
// accumulation order exactly.
func naiveGaussianBlur(src *image.NRGBA) *image.NRGBA {
	kernel := gaussianPyramidWeights()
	w := src.Bounds().Dx()
	h := src.Bounds().Dy()
	dst := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			var r, g, b, a float64
			k := 0
			for dy := -1; dy <= 1; dy++ {
				for dx := -1; dx <= 1; dx++ {
					sx := x + dx
					sy := y + dy
					if sx < 0 {
						sx = 0
					} else if sx > w-1 {
						sx = w - 1
					}
					if sy < 0 {
						sy = 0
					} else if sy > h-1 {
						sy = h - 1
					}
					s := getPix(src, sx, sy)
					aw := float64(s.A) * kernel[k]
					r += float64(s.R) * aw
					g += float64(s.G) * aw
					b += float64(s.B) * aw
					a += aw
					k++
				}
			}
			o := y*dst.Stride + x*4
			if a != 0 {
				dst.Pix[o+0] = clamp(r / a)
				dst.Pix[o+1] = clamp(g / a)
				dst.Pix[o+2] = clamp(b / a)
			}
			dst.Pix[o+3] = clamp(a)
		}
	}
	return dst
}

// naiveDecompose builds a Laplacian decomposition using only the naive blur and
// the package's documented Box downsampling/upsampling via Resize.
func naiveDecompose(t *testing.T, img image.Image, levels int) *LaplacianDecomposition {
	t.Helper()
	sizes, err := PyramidSizes(img.Bounds().Dx(), img.Bounds().Dy(), levels)
	if err != nil {
		t.Fatal(err)
	}
	cur := Clone(img)
	dec := &LaplacianDecomposition{}
	for i := 0; i < levels-1; i++ {
		blurred := naiveGaussianBlur(cur)
		next := Resize(blurred, sizes[i+1].X, sizes[i+1].Y, Box)
		predictor := Resize(next, sizes[i].X, sizes[i].Y, Box)
		w := sizes[i].X
		h := sizes[i].Y
		layer := &LaplacianLayer{
			Rect:   image.Rect(0, 0, w, h),
			Stride: w * 4,
			Pix:    make([]int16, w*h*4),
		}
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				s := getPix(cur, x, y)
				p := getPix(predictor, x, y)
				o := y*layer.Stride + x*4
				layer.Pix[o+0] = int16(s.R) - int16(p.R)
				layer.Pix[o+1] = int16(s.G) - int16(p.G)
				layer.Pix[o+2] = int16(s.B) - int16(p.B)
				layer.Pix[o+3] = int16(s.A) - int16(p.A)
			}
		}
		dec.Details = append(dec.Details, layer)
		cur = next
	}
	dec.Base = cur
	return dec
}

func equalLaplacianLayers(a, b *LaplacianLayer) string {
	if a.Rect != b.Rect || a.Stride != b.Stride || len(a.Pix) != len(b.Pix) {
		return "shape mismatch"
	}
	for i := range a.Pix {
		if a.Pix[i] != b.Pix[i] {
			x := (i % a.Stride) / 4
			y := i / a.Stride
			return "layer sample differs at (" + itoa(x) + "," + itoa(y) + ")"
		}
	}
	return ""
}

// TestDecomposeMatchesNaiveReference compares every retained sample, level by
// level, with a naive reference implementation.
func TestDecomposeMatchesNaiveReference(t *testing.T) {
	sizes := []struct{ w, h int }{
		{1, 1}, {3, 3}, {5, 1}, {1, 5}, {9, 6}, {11, 7}, {16, 9}, {17, 13},
	}
	for _, sz := range sizes {
		src := gradientNRGBA(sz.w, sz.h)
		levels := maxPyramidLevels(sz.w, sz.h)
		got, err := DecomposeLaplacian(src, levels)
		if err != nil {
			t.Fatalf("%dx%d: %v", sz.w, sz.h, err)
		}
		want := naiveDecompose(t, src, levels)
		if len(got.Details) != len(want.Details) {
			t.Fatalf("%dx%d: detail count %d want %d", sz.w, sz.h, len(got.Details), len(want.Details))
		}
		for i := range got.Details {
			if d := equalLaplacianLayers(got.Details[i], want.Details[i]); d != "" {
				t.Fatalf("%dx%d detail %d: %s", sz.w, sz.h, i, d)
			}
		}
		if d := pixelsEqual(got.Base, want.Base); d != "" {
			t.Fatalf("%dx%d base: %s", sz.w, sz.h, d)
		}
	}
}

// TestDecomposeGaussianLevelsMatch verifies the Gaussian builder and the
// Laplacian intermediate levels agree (same analysis path).
func TestDecomposeGaussianLevelsMatch(t *testing.T) {
	src := gradientNRGBA(13, 9)
	levels := 4
	gauss, err := BuildGaussianPyramid(src, levels)
	if err != nil {
		t.Fatal(err)
	}
	dec, err := DecomposeLaplacian(src, levels)
	if err != nil {
		t.Fatal(err)
	}
	// Reconstruct the Gaussian levels from the decomposition.
	rebuilt := make([]*image.NRGBA, levels)
	rebuilt[levels-1] = dec.Base
	cur := dec.Base
	for i := levels - 2; i >= 0; i-- {
		pred := Resize(cur, dec.Details[i].Rect.Dx(), dec.Details[i].Rect.Dy(), Box)
		cur = addLaplacianLayer(pred, dec.Details[i])
		rebuilt[i] = cur
	}
	for i := 0; i < levels; i++ {
		if d := pixelsEqual(rebuilt[i], gauss[i]); d != "" {
			t.Fatalf("level %d: %s", i, d)
		}
	}
}

// -----------------------------------------------------------------------------
// Exact round trip
// -----------------------------------------------------------------------------

func assertRoundTrip(t *testing.T, name string, src image.Image) {
	t.Helper()
	w := src.Bounds().Dx()
	h := src.Bounds().Dy()
	max := maxPyramidLevels(w, h)
	for levels := 1; levels <= max; levels++ {
		dec, err := DecomposeLaplacian(src, levels)
		if err != nil {
			t.Fatalf("%s levels=%d: %v", name, levels, err)
		}
		if dec.Levels() != levels {
			t.Fatalf("%s levels=%d: Levels()=%d", name, levels, dec.Levels())
		}
		got, err := dec.Reconstruct()
		if err != nil {
			t.Fatalf("%s levels=%d: reconstruct: %v", name, levels, err)
		}
		want := Clone(src)
		// Reconstruct restores the exact source bounds (including a non-zero
		// origin) and the source size.
		if got.Bounds() != src.Bounds() {
			t.Fatalf("%s levels=%d: reconstructed bounds %v want %v", name, levels, got.Bounds(), src.Bounds())
		}
		if got.Bounds().Dx() != w || got.Bounds().Dy() != h {
			t.Fatalf("%s levels=%d: reconstructed size %dx%d want %dx%d", name, levels, got.Bounds().Dx(), got.Bounds().Dy(), w, h)
		}
		// Zero per-channel error, asserted sample-by-sample.
		if diff := sameContent(got, want); diff != "" {
			t.Fatalf("%s levels=%d: %s", name, levels, diff)
		}
	}
}

func TestRoundTripOddSizes(t *testing.T) {
	for _, sz := range []struct{ w, h int }{
		{1, 1}, {1, 2}, {2, 1}, {1, 7}, {7, 1}, {2, 2}, {3, 3},
		{5, 4}, {7, 5}, {9, 6}, {11, 9}, {15, 17}, {17, 15}, {31, 33},
	} {
		assertRoundTrip(t, "odd "+itoa(sz.w)+"x"+itoa(sz.h), gradientNRGBA(sz.w, sz.h))
	}
}

func TestRoundTripNonZeroBoundsAndStride(t *testing.T) {
	big, sub := paddedSubImage(11, 9)
	before := cloneBytes(big)
	assertRoundTrip(t, "subimage", sub)
	for i := range before {
		if big.Pix[i] != before[i] {
			t.Fatalf("source modified at byte %d", i)
		}
	}
}

// TestSourceBoundsRestored covers the SourceBounds contract explicitly.
func TestSourceBoundsRestored(t *testing.T) {
	_, sub := paddedSubImage(7, 5)
	dec, err := DecomposeLaplacian(sub, 3)
	if err != nil {
		t.Fatal(err)
	}
	if dec.SourceBounds != sub.Bounds() {
		t.Fatalf("SourceBounds = %v want %v", dec.SourceBounds, sub.Bounds())
	}
	if dec.Bounds() != sub.Bounds() {
		t.Fatalf("Bounds() = %v want %v", dec.Bounds(), sub.Bounds())
	}
	if dec.Details[0].Rect != image.Rect(0, 0, 7, 5) {
		t.Fatalf("detail rect = %v want origin-normalized", dec.Details[0].Rect)
	}
	got, err := dec.Reconstruct()
	if err != nil {
		t.Fatal(err)
	}
	if got.Bounds() != sub.Bounds() {
		t.Fatalf("reconstructed bounds = %v want %v", got.Bounds(), sub.Bounds())
	}
	// The same pixel in the original frame must hold the same value.
	if p1, p2 := getPix(got, sub.Bounds().Min.X, sub.Bounds().Min.Y), getPix(Clone(sub), 0, 0); p1 != p2 {
		t.Fatalf("origin pixel = %v want %v", p1, p2)
	}

	// A hand-built decomposition without SourceBounds stays origin-normalized.
	layer := &LaplacianLayer{Rect: image.Rect(0, 0, 2, 2), Stride: 8, Pix: make([]int16, 16)}
	manual := &LaplacianDecomposition{Base: mustNRGBA(1, 1), Details: []*LaplacianLayer{layer}}
	out, err := manual.Reconstruct()
	if err != nil {
		t.Fatal(err)
	}
	if out.Bounds() != image.Rect(0, 0, 2, 2) {
		t.Fatalf("manual reconstruct bounds = %v", out.Bounds())
	}
	if manual.Bounds() != image.Rect(0, 0, 2, 2) {
		t.Fatalf("manual Bounds() = %v", manual.Bounds())
	}
}

func TestRoundTripConstant(t *testing.T) {
	for _, c := range []color.NRGBA{
		{0, 0, 0, 0},
		{255, 255, 255, 255},
		{10, 20, 30, 40},
		{200, 100, 50, 255},
	} {
		src := mustNRGBA(9, 7)
		for y := 0; y < 7; y++ {
			for x := 0; x < 9; x++ {
				setPix(src, x, y, c)
			}
		}
		assertRoundTrip(t, "constant", src)
		// For a constant image every detail sample must be zero.
		dec, err := DecomposeLaplacian(src, 4)
		if err != nil {
			t.Fatal(err)
		}
		for i, l := range dec.Details {
			for j, v := range l.Pix {
				if v != 0 {
					t.Fatalf("constant %v: detail %d sample %d = %d want 0", c, i, j, v)
				}
			}
		}
	}
}

func TestRoundTripImpulse(t *testing.T) {
	src := mustNRGBA(15, 15)
	setPix(src, 7, 7, color.NRGBA{R: 255, G: 255, B: 255, A: 255})
	assertRoundTrip(t, "impulse", src)
}

func TestRoundTripAlphaEdges(t *testing.T) {
	src := mustNRGBA(9, 7)
	for y := 0; y < 7; y++ {
		for x := 0; x < 9; x++ {
			// Visible on the left half, fully transparent with hidden chroma on
			// the right half, plus a soft alpha gradient in between.
			switch {
			case x < 3:
				setPix(src, x, y, color.NRGBA{R: uint8(200 + x), G: 40, B: 60, A: 255})
			case x < 5:
				setPix(src, x, y, color.NRGBA{R: 10, G: 220, B: 90, A: uint8(64 + x*16)})
			default:
				setPix(src, x, y, color.NRGBA{R: 255, G: 0, B: 255, A: 0})
			}
		}
	}
	assertRoundTrip(t, "alpha edges", src)
	// Hidden RGB must survive reconstruction exactly; assert the transparent
	// corner samples explicitly.
	dec, err := DecomposeLaplacian(src, 3)
	if err != nil {
		t.Fatal(err)
	}
	got, err := dec.Reconstruct()
	if err != nil {
		t.Fatal(err)
	}
	if p := getPix(got, 8, 6); p != (color.NRGBA{R: 255, G: 0, B: 255, A: 0}) {
		t.Fatalf("hidden transparent pixel = %v want {255,0,255,0}", p)
	}
	// A visible pixel must not have absorbed the hidden magenta.
	if p := getPix(got, 0, 0); p != (color.NRGBA{R: 200, G: 40, B: 60, A: 255}) {
		t.Fatalf("visible pixel = %v want {200,40,60,255}", p)
	}
}

// TestRoundTripOtherColorModels asserts reconstruction equals the package's
// existing Clone semantics for every supported color model.
func TestRoundTripOtherColorModels(t *testing.T) {
	b := image.Rect(0, 0, 10, 8)

	rgba := image.NewRGBA(b)
	gray := image.NewGray(b)
	gray16 := image.NewGray16(b)
	ycbr := image.NewYCbCr(b, image.YCbCrSubsampleRatio420)
	pal := image.NewPaletted(b, color.Palette{
		color.NRGBA{0, 0, 0, 0},
		color.NRGBA{255, 0, 0, 255},
		color.NRGBA{0, 255, 0, 128},
		color.NRGBA{10, 20, 30, 200},
	})
	for y := 0; y < 8; y++ {
		for x := 0; x < 10; x++ {
			rgba.Set(x, y, color.NRGBA{uint8(x * 20), uint8(y * 25), uint8(x*7 + y*3), uint8(255 - x*10)})
			gray.SetGray(x, y, color.Gray{uint8(x*11 + y*29)})
			gray16.SetGray16(x, y, color.Gray16{uint16(x*5000 + y*811 + 7)})
		}
	}
	for i := range ycbr.Y {
		ycbr.Y[i] = uint8(i * 3)
	}
	for i := range ycbr.Cb {
		ycbr.Cb[i] = uint8(100 + i)
		ycbr.Cr[i] = uint8(150 - i)
	}
	for y := 0; y < 8; y++ {
		for x := 0; x < 10; x++ {
			pal.SetColorIndex(x, y, uint8((x+y)%4))
		}
	}

	for name, src := range map[string]image.Image{
		"RGBA": rgba, "Gray": gray, "Gray16": gray16, "YCbCr": ycbr, "Paletted": pal,
	} {
		assertRoundTrip(t, name, src)
	}
}

// TestRoundTripGray16HighByte documents the existing 16-bit mapping: the high
// byte is used, exactly as newScanner does.
func TestRoundTripGray16HighByte(t *testing.T) {
	src := image.NewGray16(image.Rect(0, 0, 4, 4))
	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			src.SetGray16(x, y, color.Gray16{uint16(x*0x1234 + y*0x0101)})
		}
	}
	levels, err := BuildGaussianPyramid(src, 3)
	if err != nil {
		t.Fatal(err)
	}
	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			want := uint8(src.Gray16At(x, y).Y >> 8)
			got := getPix(levels[0], x, y)
			if got.R != want || got.G != want || got.B != want || got.A != 255 {
				t.Fatalf("(%d,%d) = %v want gray %d", x, y, got, want)
			}
		}
	}
}

// TestReconstructInvalid verifies the stable error for inconsistent input.
func TestReconstructInvalid(t *testing.T) {
	var nilDec *LaplacianDecomposition
	if _, err := nilDec.Reconstruct(); !errors.Is(err, ErrInvalidDecomposition) {
		t.Errorf("nil decomposition: got %v", err)
	}
	if _, err := (&LaplacianDecomposition{}).Reconstruct(); !errors.Is(err, ErrInvalidDecomposition) {
		t.Errorf("empty decomposition: got %v", err)
	}

	valid, err := DecomposeLaplacian(gradientNRGBA(8, 6), 3)
	if err != nil {
		t.Fatal(err)
	}
	bad := &LaplacianDecomposition{Base: valid.Base, Details: []*LaplacianLayer{nil}}
	if _, err := bad.Reconstruct(); !errors.Is(err, ErrInvalidDecomposition) {
		t.Errorf("nil layer: got %v", err)
	}
	bad = &LaplacianDecomposition{Base: valid.Base, Details: []*LaplacianLayer{{
		Rect: image.Rect(0, 0, 8, 6), Stride: 4 * 8, Pix: make([]int16, 3),
	}}}
	if _, err := bad.Reconstruct(); !errors.Is(err, ErrInvalidDecomposition) {
		t.Errorf("short pix: got %v", err)
	}
	bad = &LaplacianDecomposition{Base: valid.Base, Details: []*LaplacianLayer{{
		Rect: image.Rect(0, 0, 9, 6), Stride: 4 * 9, Pix: make([]int16, 9*6*4),
	}}}
	if _, err := bad.Reconstruct(); !errors.Is(err, ErrInvalidDecomposition) {
		t.Errorf("size mismatch: got %v", err)
	}
	// Saturated reconstruction of a hand-built layer stays in range.
	layer := &LaplacianLayer{Rect: image.Rect(0, 0, 1, 1), Stride: 4, Pix: []int16{1000, -1000, 500, -500}}
	out, err := (&LaplacianDecomposition{Base: mustNRGBA(1, 1), Details: []*LaplacianLayer{layer}}).Reconstruct()
	if err != nil {
		t.Fatal(err)
	}
	if out.Bounds() != image.Rect(0, 0, 1, 1) {
		t.Fatalf("bounds = %v", out.Bounds())
	}
}

// TestReconstructDoesNotMutate checks that reconstruction leaves the
// decomposition intact and is repeatable.
func TestReconstructDoesNotMutate(t *testing.T) {
	src := gradientNRGBA(10, 7)
	dec, err := DecomposeLaplacian(src, 3)
	if err != nil {
		t.Fatal(err)
	}
	beforePix := cloneBytes(dec.Base)
	detailCopy := make([][]int16, len(dec.Details))
	for i, l := range dec.Details {
		detailCopy[i] = append([]int16(nil), l.Pix...)
	}
	first, err := dec.Reconstruct()
	if err != nil {
		t.Fatal(err)
	}
	second, err := dec.Reconstruct()
	if err != nil {
		t.Fatal(err)
	}
	if diff := pixelsEqual(first, second); diff != "" {
		t.Fatalf("repeated reconstruct: %s", diff)
	}
	if diff := pixelsEqual(dec.Base, &image.NRGBA{Pix: beforePix, Stride: dec.Base.Stride, Rect: dec.Base.Rect}); diff != "" {
		t.Fatalf("base mutated: %s", diff)
	}
	for i, l := range dec.Details {
		for j := range l.Pix {
			if l.Pix[j] != detailCopy[i][j] {
				t.Fatalf("detail %d mutated at %d", i, j)
			}
		}
	}
}

// -----------------------------------------------------------------------------
// Repeatability and concurrency
// -----------------------------------------------------------------------------

func TestRepeatedCallsDeterministic(t *testing.T) {
	src := gradientNRGBA(23, 17)
	a, err := DecomposeLaplacian(src, 5)
	if err != nil {
		t.Fatal(err)
	}
	b, err := DecomposeLaplacian(src, 5)
	if err != nil {
		t.Fatal(err)
	}
	if len(a.Details) != len(b.Details) {
		t.Fatal("level count differs")
	}
	for i := range a.Details {
		if d := equalLaplacianLayers(a.Details[i], b.Details[i]); d != "" {
			t.Fatalf("detail %d: %s", i, d)
		}
	}
	if d := pixelsEqual(a.Base, b.Base); d != "" {
		t.Fatalf("base: %s", d)
	}
}

// TestSingleVsMultiThreaded compares results computed with one worker against
// the default worker pool, and runs many concurrent builds to exercise the race
// detector.
func TestSingleVsMultiThreaded(t *testing.T) {
	src := gradientNRGBA(37, 29)

	SetMaxProcs(1)
	single, err := BuildGaussianPyramid(src, 6)
	if err != nil {
		t.Fatal(err)
	}
	singleDec, err := DecomposeLaplacian(src, 6)
	if err != nil {
		t.Fatal(err)
	}
	SetMaxProcs(0)

	multi, err := BuildGaussianPyramid(src, 6)
	if err != nil {
		t.Fatal(err)
	}
	multiDec, err := DecomposeLaplacian(src, 6)
	if err != nil {
		t.Fatal(err)
	}
	for i := range single {
		if d := pixelsEqual(single[i], multi[i]); d != "" {
			t.Fatalf("level %d single vs multi: %s", i, d)
		}
	}
	for i := range singleDec.Details {
		if d := equalLaplacianLayers(singleDec.Details[i], multiDec.Details[i]); d != "" {
			t.Fatalf("detail %d single vs multi: %s", i, d)
		}
	}

	// Concurrent independent builds must match the serial result.
	const workers = 8
	var wg sync.WaitGroup
	errs := make(chan string, workers)
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			levels, err := BuildGaussianPyramid(src, 6)
			if err != nil {
				errs <- err.Error()
				return
			}
			for j := range multi {
				if d := pixelsEqual(levels[j], multi[j]); d != "" {
					errs <- "concurrent level mismatch: " + d
					return
				}
			}
			dec, err := DecomposeLaplacian(src, 6)
			if err != nil {
				errs <- err.Error()
				return
			}
			got, err := dec.Reconstruct()
			if err != nil {
				errs <- err.Error()
				return
			}
			if d := pixelsEqual(got, Clone(src)); d != "" {
				errs <- "concurrent round trip: " + d
			}
		}()
	}
	wg.Wait()
	close(errs)
	for e := range errs {
		t.Fatal(e)
	}
}

// TestConcurrentReconstruct reconstructs one immutable decomposition from many
// goroutines.
func TestConcurrentReconstruct(t *testing.T) {
	src := gradientNRGBA(21, 15)
	dec, err := DecomposeLaplacian(src, 4)
	if err != nil {
		t.Fatal(err)
	}
	want := Clone(src)
	var wg sync.WaitGroup
	errs := make(chan string, 8)
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			got, err := dec.Reconstruct()
			if err != nil {
				errs <- err.Error()
				return
			}
			if d := pixelsEqual(got, want); d != "" {
				errs <- d
			}
		}()
	}
	wg.Wait()
	close(errs)
	for e := range errs {
		t.Fatal(e)
	}
}

// -----------------------------------------------------------------------------
// Benchmarks: time and memory growth per level
// -----------------------------------------------------------------------------

func benchmarkPyramid(b *testing.B, size int) {
	src := gradientNRGBA(size, size)
	levels := 5
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := BuildGaussianPyramid(src, levels); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkGaussianPyramid64(b *testing.B)  { benchmarkPyramid(b, 64) }
func BenchmarkGaussianPyramid128(b *testing.B) { benchmarkPyramid(b, 128) }
func BenchmarkGaussianPyramid256(b *testing.B) { benchmarkPyramid(b, 256) }
func BenchmarkGaussianPyramid512(b *testing.B) { benchmarkPyramid(b, 512) }

func BenchmarkLaplacianDecompose256(b *testing.B) {
	src := gradientNRGBA(256, 256)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := DecomposeLaplacian(src, 5); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkLaplacianReconstruct256(b *testing.B) {
	dec, err := DecomposeLaplacian(gradientNRGBA(256, 256), 5)
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := dec.Reconstruct(); err != nil {
			b.Fatal(err)
		}
	}
}
