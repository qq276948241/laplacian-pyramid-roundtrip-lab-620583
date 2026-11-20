package imaging

import (
	"context"
	"errors"
	"image"
)

// Multi-scale image decomposition.
//
// This file adds a reversible Gaussian/Laplacian pyramid on top of the existing
// image, convolution, and resize semantics. Existing filter entry points
// (Resize, Convolve3x3, Blur, ...) are not modified.
//
// # Coordinate convention
//
// Every returned level has bounds image.Rect(0, 0, w, h). Input images may have
// arbitrary bounds, including a non-zero origin and a padded stride (for
// example a SubImage); source coordinates are read relative to img.Bounds().Min.
// LaplacianDecomposition records the source bounds in SourceBounds, so
// Reconstruct returns an image whose bounds equal img.Bounds() when the source
// origin is non-zero. Without that field (a hand-built decomposition), the
// result is origin-normalized and its size equals the source Bounds().Size().
// Gaussian pyramid levels are always origin-normalized; add the source minimum
// to a level rectangle if the original coordinate frame is needed.
//
// # Level sizes and downsampling
//
// Level 0 is the source. For every following level:
//
//	w(k+1) = ceil(w(k)/2) = w(k)/2 + w(k)%2   (minimum 1)
//	h(k+1) = ceil(h(k)/2) = h(k)/2 + h(k)%2   (minimum 1)
//
// Odd rows and columns are therefore never dropped: ceil rounding folds them
// into the last output row/column through the resampling weights. Let
// maxLevels(w, h) be the number of levels that fit before both dimensions reach
// 1 pixel. Requesting more levels returns ErrTooManyLevels.
//
// # Analysis kernels and edge rules
//
// Before every downsample step the current level is filtered with the fixed
// normalized 3x3 Gaussian kernel
//
//	1/16 2/16 1/16
//	2/16 4/16 2/16
//	1/16 2/16 1/16
//
// Filtering is done in premultiplied alpha space (each RGB sample is weighted
// by alpha), so RGB values hidden under fully transparent pixels never bleed
// into visible regions. Borders use replication: out-of-range samples are
// replaced by the nearest edge sample, and because the weights sum to exactly
// 1.0 no edge renormalization is performed. Samples are accumulated in
// row-major tap order, then rounded with round-half-up (the package's existing
// clamp: floor(x+0.5)) and saturated to [0, 255].
//
// Downsampling and the reconstruction predictor use Resize with the Box
// resampling filter (support 0.5), preserving the existing Resize semantics:
// indices are sampled relative to the image origin, the source window is
// clamped to valid indices, and weights are normalized over the valid taps.
// Resize also filters alpha premultiplied, so it has the same no-leak behavior.
//
// # Laplacian layers and exact reconstruction
//
// For Gaussian level G(k) and predictor P(k) = Resize(G(k+1), size(G(k)), Box),
// the detail layer stores the signed sample difference
//
//	D(k)[c](x, y) = int16(G(k)[c](x, y)) - int16(P(k)[c](x, y))
//
// for every channel c in {R, G, B, A}. Differences lie in [-255, 255] and are
// stored losslessly in int16 storage, so reconstruction
//
//	G(k) = P(k) + D(k)
//
// is bit-exact: the maximum per-sample error is 0 for every color and alpha
// channel. Reconstruction still saturates the integer sum to [0, 255] as a
// defensive measure; for layers produced by this package the sum is already a
// valid uint8 value and saturation never changes it.
//
// No package-level mutable buffers are used. Every call allocates its own
// buffers, and row buffers are allocated inside the worker goroutines used by
// parallel, so concurrent decompositions cannot share mutable state.

// Errors returned by the decomposition API. They are sentinels and can be
// compared with == or tested with errors.Is.
var (
	// ErrNilImage means a nil image was passed to a decomposition function.
	ErrNilImage = errors.New("imaging: nil image")

	// ErrEmptyImage means the image has empty or non-positive bounds.
	ErrEmptyImage = errors.New("imaging: empty image")

	// ErrInvalidLevels means the requested level count is less than 1.
	ErrInvalidLevels = errors.New("imaging: invalid level count")

	// ErrTooManyLevels means the requested level count exceeds the number of
	// times the image can be downsampled before reaching 1x1.
	ErrTooManyLevels = errors.New("imaging: too many pyramid levels for image size")

	// ErrImageTooLarge means the image or its pyramid would overflow the
	// per-call allocation budget. No large buffer is allocated in this case.
	ErrImageTooLarge = errors.New("imaging: image exceeds decomposition budget")

	// ErrInvalidDecomposition means a LaplacianDecomposition was nil or
	// internally inconsistent and cannot be reconstructed.
	ErrInvalidDecomposition = errors.New("imaging: invalid laplacian decomposition")
)

// MaxDecompositionBytes is the upper bound on the bytes a single decomposition
// call may allocate for image data. It is checked against the full planned
// pyramid (all retained levels plus the largest temporary buffers) before any
// large allocation is made. Calls that would exceed it return ErrImageTooLarge.
const MaxDecompositionBytes int64 = 1 << 30 // 1 GiB

// gaussianPyramidWeights returns the fixed, normalized 3x3 Gaussian analysis
// kernel. It returns a copy by value so no package-level mutable state is
// shared between concurrent decompositions. Every value is an exact binary
// fraction, so the weights sum to exactly 1.0 in float64 and the filter neither
// darkens nor brightens a constant image.
func gaussianPyramidWeights() [9]float64 {
	return [9]float64{
		1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0,
		2.0 / 16.0, 4.0 / 16.0, 2.0 / 16.0,
		1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0,
	}
}

// LaplacianLayer is one signed Laplacian detail level.
//
// Pix holds four int16 samples per pixel in R, G, B, A order, using the same
// non-premultiplied 8-bit channel space as image.NRGBA. Each sample is the
// difference between the Gaussian level and its upsampled predictor and lies in
// [-255, 255]. Rect is always image.Rect(0, 0, w, h); Stride is the number of
// int16 samples per row (4*width for layers produced by this package).
type LaplacianLayer struct {
	Rect   image.Rectangle
	Stride int
	Pix    []int16
}

// Bounds returns the layer bounds.
func (l *LaplacianLayer) Bounds() image.Rectangle {
	if l == nil {
		return image.Rectangle{}
	}
	return l.Rect
}

// LaplacianDecomposition is a reversible multi-scale decomposition.
//
// Details[i] is the Laplacian layer for Gaussian level i, for
// i in [0, len(Details)-1]. Base is the final Gaussian level. The number of
// levels is len(Details)+1; level 0 has the source size and the last level has
// the smallest size. Detail layers are origin-normalized as described in the
// package documentation for this API.
//
// SourceBounds records img.Bounds() as passed to DecomposeLaplacian. It is empty
// for hand-built decompositions, in which case Reconstruct returns an
// origin-normalized image.
//
// A LaplacianDecomposition is immutable once returned and may be reconstructed
// concurrently.
type LaplacianDecomposition struct {
	Details      []*LaplacianLayer
	Base         *image.NRGBA
	SourceBounds image.Rectangle
}

// Levels returns the number of levels in the decomposition, including the
// original-size level 0.
func (d *LaplacianDecomposition) Levels() int {
	if d == nil {
		return 0
	}
	return len(d.Details) + 1
}

// Bounds returns the original source bounds when the decomposition was produced
// by DecomposeLaplacian, otherwise the bounds of level 0. A nil decomposition
// returns an empty rectangle.
func (d *LaplacianDecomposition) Bounds() image.Rectangle {
	if d == nil {
		return image.Rectangle{}
	}
	if !d.SourceBounds.Empty() {
		return d.SourceBounds
	}
	if len(d.Details) > 0 && d.Details[0] != nil {
		return d.Details[0].Rect
	}
	if d.Base != nil {
		b := d.Base.Bounds()
		return image.Rect(0, 0, b.Dx(), b.Dy())
	}
	return image.Rectangle{}
}

// PyramidSizes returns the per-level dimensions for a Gaussian or Laplacian
// pyramid whose level 0 has the given dimensions. The first element is always
// image.Pt(width, height) and each following element is ceil(previous/2) with a
// minimum of 1 in each dimension. It validates the dimensions and the level
// count exactly as the builders do, but does not require any image data.
func PyramidSizes(width, height, levels int) ([]image.Point, error) {
	if width <= 0 || height <= 0 {
		return nil, ErrEmptyImage
	}
	return pyramidSizes(width, height, levels)
}

func pyramidSizes(width, height, levels int) ([]image.Point, error) {
	if levels < 1 {
		return nil, ErrInvalidLevels
	}
	maxLevels := maxPyramidLevels(width, height)
	if levels > maxLevels {
		return nil, ErrTooManyLevels
	}
	sizes := make([]image.Point, levels)
	sizes[0] = image.Pt(width, height)
	for i := 1; i < levels; i++ {
		sizes[i] = image.Pt(halfUp(sizes[i-1].X), halfUp(sizes[i-1].Y))
	}
	return sizes, nil
}

// halfUp returns ceil(n/2) without overflowing for n == MaxInt.
func halfUp(n int) int {
	if n <= 1 {
		return 1
	}
	return n/2 + n%2
}

// maxPyramidLevels returns the largest valid level count (including level 0)
// before both dimensions have reached 1.
func maxPyramidLevels(width, height int) int {
	levels := 1
	for width > 1 || height > 1 {
		width = halfUp(width)
		height = halfUp(height)
		levels++
	}
	return levels
}

// validateSize checks image bounds and returns the origin-normalized size.
func validateSize(img image.Image) (int, int, error) {
	if img == nil {
		return 0, 0, ErrNilImage
	}
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	if w <= 0 || h <= 0 {
		return 0, 0, ErrEmptyImage
	}
	return w, h, nil
}

// pixelBytes returns w*h*bytesPerPixel after checking that the product cannot
// overflow and would not exceed MaxDecompositionBytes. The division-based check
// means even pathological int dimensions are rejected rather than wrapping.
func pixelBytes(w, h int, bytesPerPixel int64) (int64, error) {
	if w <= 0 || h <= 0 {
		return 0, ErrEmptyImage
	}
	maxPixels := MaxDecompositionBytes / bytesPerPixel
	if int64(w) > maxPixels/int64(h) {
		return 0, ErrImageTooLarge
	}
	return int64(w) * int64(h) * bytesPerPixel, nil
}

// addBytes adds n to total, returning ErrImageTooLarge on overflow or when the
// budget would be exceeded. n is always a non-negative value already checked by
// pixelBytes, so overflow can only occur from summing many levels.
func addBytes(total, n int64) (int64, error) {
	if n < 0 || total > MaxDecompositionBytes-n {
		return 0, ErrImageTooLarge
	}
	return total + n, nil
}

// checkGaussianBudget validates the plan for a Gaussian pyramid. It accounts for
// all retained levels plus the blur temporary.
func checkGaussianBudget(sizes []image.Point) error {
	baseBytes, err := pixelBytes(sizes[0].X, sizes[0].Y, 4)
	if err != nil {
		return err
	}
	var total int64
	for _, s := range sizes {
		n, err := pixelBytes(s.X, s.Y, 4)
		if err != nil {
			return err
		}
		if total, err = addBytes(total, n); err != nil {
			return err
		}
	}
	// Blur temporary at level 0 lives at the same time as the output levels.
	_, err = addBytes(total, baseBytes)
	return err
}

// checkLaplacianBudget validates the plan for a Laplacian decomposition. It
// accounts for the retained detail layers (8 bytes per pixel), the base level,
// and the temporary blurred/predictor/next buffers.
func checkLaplacianBudget(sizes []image.Point) error {
	baseBytes, err := pixelBytes(sizes[0].X, sizes[0].Y, 4)
	if err != nil {
		return err
	}
	var total int64
	for i := 0; i < len(sizes)-1; i++ {
		n, err := pixelBytes(sizes[i].X, sizes[i].Y, 8)
		if err != nil {
			return err
		}
		if total, err = addBytes(total, n); err != nil {
			return err
		}
	}
	base := sizes[len(sizes)-1]
	n, err := pixelBytes(base.X, base.Y, 4)
	if err != nil {
		return err
	}
	if total, err = addBytes(total, n); err != nil {
		return err
	}
	// Temporaries: source clone, blurred level and predictor at level 0, plus
	// the next Gaussian level.
	for i := 0; i < 3; i++ {
		if total, err = addBytes(total, baseBytes); err != nil {
			return err
		}
	}
	if len(sizes) > 1 {
		n, err := pixelBytes(sizes[1].X, sizes[1].Y, 4)
		if err != nil {
			return err
		}
		if _, err = addBytes(total, n); err != nil {
			return err
		}
	}
	return nil
}

// BuildGaussianPyramid builds a Gaussian pyramid from img. levels includes
// level 0 (a copy of the source), so levels must be at least 1 and at most the
// number of available downsample steps (see PyramidSizes).
//
// Each level is an *image.NRGBA with bounds image.Rect(0, 0, w, h). The source
// image is never written to. See the package documentation at the top of
// pyramid.go for the fixed kernel, edge extension, premultiplied alpha,
// rounding, saturation, and size rules.
//
// On error the returned slice is nil; no partial pyramid is returned.
func BuildGaussianPyramid(img image.Image, levels int) ([]*image.NRGBA, error) {
	return BuildGaussianPyramidContext(context.Background(), img, levels)
}

// BuildGaussianPyramidContext is BuildGaussianPyramid with cancellation.
// Cancellation is checked before cloning, between levels, and during row
// processing. On cancellation or any other error it returns nil and the
// context's error; partial levels are never returned.
func BuildGaussianPyramidContext(ctx context.Context, img image.Image, levels int) ([]*image.NRGBA, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	w, h, err := validateSize(img)
	if err != nil {
		return nil, err
	}
	sizes, err := pyramidSizes(w, h, levels)
	if err != nil {
		return nil, err
	}
	if err := checkGaussianBudget(sizes); err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	level0, err := cloneContext(ctx, img)
	if err != nil {
		return nil, err
	}

	out := make([]*image.NRGBA, levels)
	out[0] = level0
	cur := level0
	for i := 1; i < levels; i++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		blurred, err := gaussianBlur3x3Context(ctx, cur)
		if err != nil {
			return nil, err
		}
		next := Resize(blurred, sizes[i].X, sizes[i].Y, Box)
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if next.Bounds().Empty() {
			return nil, ErrImageTooLarge
		}
		out[i] = next
		cur = next
	}
	return out, nil
}

// DecomposeLaplacian decomposes img into Laplacian detail layers plus a
// Gaussian base. levels counts the source level, so levels-1 detail layers are
// produced. See BuildGaussianPyramid for size, kernel, alpha, rounding, and
// error rules.
//
// The decomposition is reversible: LaplacianDecomposition.Reconstruct restores
// the source sample-for-sample (zero per-channel error) with origin-normalized
// bounds. The source image is never written to. On error the returned
// decomposition is nil; partial layers are never returned.
func DecomposeLaplacian(img image.Image, levels int) (*LaplacianDecomposition, error) {
	return DecomposeLaplacianContext(context.Background(), img, levels)
}

// DecomposeLaplacianContext is DecomposeLaplacian with cancellation.
// Cancellation is checked before cloning, between levels, and during row
// processing. On cancellation or any other error it returns nil and the
// context's error.
func DecomposeLaplacianContext(ctx context.Context, img image.Image, levels int) (*LaplacianDecomposition, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	w, h, err := validateSize(img)
	if err != nil {
		return nil, err
	}
	sizes, err := pyramidSizes(w, h, levels)
	if err != nil {
		return nil, err
	}
	if err := checkLaplacianBudget(sizes); err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	cur, err := cloneContext(ctx, img)
	if err != nil {
		return nil, err
	}
	sourceBounds := img.Bounds()

	details := make([]*LaplacianLayer, 0, levels-1)
	for i := 0; i < levels-1; i++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}

		blurred, err := gaussianBlur3x3Context(ctx, cur)
		if err != nil {
			return nil, err
		}
		next := Resize(blurred, sizes[i+1].X, sizes[i+1].Y, Box)
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if next.Bounds().Empty() {
			return nil, ErrImageTooLarge
		}

		predictor := Resize(next, sizes[i].X, sizes[i].Y, Box)
		if predictor.Bounds().Empty() {
			return nil, ErrImageTooLarge
		}

		details = append(details, newLaplacianLayer(cur, predictor))
		cur = next
	}

	return &LaplacianDecomposition{Details: details, Base: cur, SourceBounds: sourceBounds}, nil
}

// Reconstruct rebuilds the image from the decomposition. For any decomposition
// produced by DecomposeLaplacian the result is bit-exact: every R, G, B and A
// sample equals the source level 0 sample, including RGB values stored under
// fully transparent pixels.
//
// If SourceBounds is non-empty (as set by DecomposeLaplacian) the returned image
// has exactly those bounds, so a source with a non-zero origin is restored in
// its original coordinate frame. Otherwise the result is origin-normalized.
//
// Reconstruct validates the decomposition and returns ErrInvalidDecomposition
// for nil or internally inconsistent data. It does not modify the
// decomposition, and concurrent calls are safe.
func (d *LaplacianDecomposition) Reconstruct() (*image.NRGBA, error) {
	if d == nil || d.Base == nil {
		return nil, ErrInvalidDecomposition
	}
	bw, bh, err := validateSize(d.Base)
	if err != nil {
		return nil, ErrInvalidDecomposition
	}
	if _, err := pixelBytes(bw, bh, 4); err != nil {
		return nil, ErrInvalidDecomposition
	}

	cur := Clone(d.Base)
	if cur.Bounds().Empty() {
		return nil, ErrInvalidDecomposition
	}

	for i := len(d.Details) - 1; i >= 0; i-- {
		layer := d.Details[i]
		if err := validateLaplacianLayer(layer); err != nil {
			return nil, err
		}
		// Reject layers that would exceed the allocation budget before
		// Resize allocates anything.
		if _, err := pixelBytes(layer.Rect.Dx(), layer.Rect.Dy(), 4); err != nil {
			return nil, ErrInvalidDecomposition
		}
		// cur is Gaussian level i+1, so its size must be the ceil-halved size
		// of this layer. This also validates the Base size for the last layer.
		if cur.Bounds().Dx() != halfUp(layer.Rect.Dx()) || cur.Bounds().Dy() != halfUp(layer.Rect.Dy()) {
			return nil, ErrInvalidDecomposition
		}
		predictor := Resize(cur, layer.Rect.Dx(), layer.Rect.Dy(), Box)
		if predictor.Bounds() != image.Rect(0, 0, layer.Rect.Dx(), layer.Rect.Dy()) {
			return nil, ErrInvalidDecomposition
		}
		out := addLaplacianLayer(predictor, layer)
		if out == nil {
			return nil, ErrInvalidDecomposition
		}
		cur = out
	}
	if !d.SourceBounds.Empty() {
		out := image.NewNRGBA(d.SourceBounds)
		rowBytes := d.SourceBounds.Dx() * 4
		for y := 0; y < d.SourceBounds.Dy(); y++ {
			srcOff := y * cur.Stride
			dstOff := y * out.Stride
			copy(out.Pix[dstOff:dstOff+rowBytes], cur.Pix[srcOff:srcOff+rowBytes])
		}
		return out, nil
	}
	return cur, nil
}

func validateLaplacianLayer(l *LaplacianLayer) error {
	if l == nil {
		return ErrInvalidDecomposition
	}
	w, h := l.Rect.Dx(), l.Rect.Dy()
	if w <= 0 || h <= 0 {
		return ErrInvalidDecomposition
	}
	// Check the stride multiplication and the sample count with division so a
	// hostile Rectangle cannot overflow into a negative or tiny value.
	if w > int(^uint(0)>>1)/4 {
		return ErrInvalidDecomposition
	}
	if l.Stride < w*4 {
		return ErrInvalidDecomposition
	}
	if l.Stride <= 0 {
		return ErrInvalidDecomposition
	}
	if int64(h) > int64(len(l.Pix))/int64(l.Stride) {
		return ErrInvalidDecomposition
	}
	return nil
}

// newLaplacianLayer computes src-predictor for every channel.
func newLaplacianLayer(src, predictor *image.NRGBA) *LaplacianLayer {
	w := src.Bounds().Dx()
	h := src.Bounds().Dy()
	l := &LaplacianLayer{
		Rect:   image.Rect(0, 0, w, h),
		Stride: w * 4,
		Pix:    make([]int16, w*h*4),
	}
	for y := 0; y < h; y++ {
		sOff := y*src.Stride + src.Rect.Min.X*4
		pOff := y*predictor.Stride + predictor.Rect.Min.X*4
		dOff := y * l.Stride
		for x := 0; x < w; x++ {
			for c := 0; c < 4; c++ {
				l.Pix[dOff+c] = int16(src.Pix[sOff+c]) - int16(predictor.Pix[pOff+c])
			}
			sOff += 4
			pOff += 4
			dOff += 4
		}
	}
	return l
}

// addLaplacianLayer computes predictor+layer and saturates to [0, 255].
func addLaplacianLayer(predictor *image.NRGBA, l *LaplacianLayer) *image.NRGBA {
	w := l.Rect.Dx()
	h := l.Rect.Dy()
	if w <= 0 || h <= 0 {
		return nil
	}
	out := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		pOff := y*predictor.Stride + predictor.Rect.Min.X*4
		dOff := y * l.Stride
		oOff := y * out.Stride
		for x := 0; x < w; x++ {
			for c := 0; c < 4; c++ {
				v := int(predictor.Pix[pOff+c]) + int(l.Pix[dOff+c])
				if v < 0 {
					v = 0
				} else if v > 255 {
					v = 255
				}
				out.Pix[oOff+c] = uint8(v)
			}
			pOff += 4
			dOff += 4
			oOff += 4
		}
	}
	return out
}

// cloneContext copies img into a compact, origin-normalized *image.NRGBA while
// honoring cancellation. It only reads img.
func cloneContext(ctx context.Context, img image.Image) (*image.NRGBA, error) {
	src := newScanner(img)
	if src.w <= 0 || src.h <= 0 {
		return nil, ErrEmptyImage
	}
	dst := image.NewNRGBA(image.Rect(0, 0, src.w, src.h))
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	parallel(0, src.h, func(ys <-chan int) {
		row := make([]uint8, src.w*4)
		for y := range ys {
			if ctx.Err() != nil {
				return
			}
			src.scan(0, y, src.w, y+1, row)
			copy(dst.Pix[y*dst.Stride:], row)
		}
	})
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return dst, nil
}

// gaussianBlur3x3Context applies the fixed Gaussian analysis kernel in
// premultiplied alpha space. Borders are replicated, taps are accumulated in
// row-major order, results are rounded with clamp and saturated to [0, 255].
// The input is never modified. Workers write disjoint destination rows and use
// no shared scratch buffer, so concurrent calls stay independent.
func gaussianBlur3x3Context(ctx context.Context, src *image.NRGBA) (*image.NRGBA, error) {
	w := src.Bounds().Dx()
	h := src.Bounds().Dy()
	if w <= 0 || h <= 0 {
		return nil, ErrEmptyImage
	}
	dst := image.NewNRGBA(image.Rect(0, 0, w, h))
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	kernel := gaussianPyramidWeights()
	srcMinX := src.Rect.Min.X
	srcMinY := src.Rect.Min.Y
	parallel(0, h, func(ys <-chan int) {
		for y := range ys {
			if ctx.Err() != nil {
				return
			}
			for x := 0; x < w; x++ {
				var r, g, b, a float64
				k := 0
				for dy := -1; dy <= 1; dy++ {
					iy := y + dy
					if iy < 0 {
						iy = 0
					} else if iy >= h {
						iy = h - 1
					}
					for dx := -1; dx <= 1; dx++ {
						ix := x + dx
						if ix < 0 {
							ix = 0
						} else if ix >= w {
							ix = w - 1
						}
						off := (srcMinY+iy)*src.Stride + (srcMinX+ix)*4
						aw := float64(src.Pix[off+3]) * kernel[k]
						r += float64(src.Pix[off]) * aw
						g += float64(src.Pix[off+1]) * aw
						b += float64(src.Pix[off+2]) * aw
						a += aw
						k++
					}
				}
				off := y*dst.Stride + x*4
				if a != 0 {
					aInv := 1 / a
					dst.Pix[off] = clamp(r * aInv)
					dst.Pix[off+1] = clamp(g * aInv)
					dst.Pix[off+2] = clamp(b * aInv)
				}
				dst.Pix[off+3] = clamp(a)
			}
		}
	})
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return dst, nil
}
