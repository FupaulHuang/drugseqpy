# drugseqpy logo

A four-well culture plate with a clipped orientation corner. Three teal wells
and one gold well suggest a treatment response; the navy outline and lowercase
wordmark keep the design legible at small sizes.

| Asset | SVG | Transparent PNG |
| --- | --- | --- |
| Primary wordmark | [SVG](drugseqpy-logo.svg) | [PNG](drugseqpy-logo.png) |
| Plate icon | [SVG](drugseqpy-icon.svg) | [PNG](drugseqpy-icon.png) |
| Monochrome wordmark | [SVG](drugseqpy-logo-monochrome.svg) | [PNG](drugseqpy-logo-monochrome.png) |
| Wordmark for dark backgrounds | [SVG](drugseqpy-logo-dark.svg) | [PNG](drugseqpy-logo-dark.png) |

SVGs contain vector paths, including the lettering, so no font installation
is needed to display them. PNG wordmarks are 2400 × 768 pixels; the icon is
768 × 768 pixels. All logo exports have transparent backgrounds.

Colors: navy `#17384A`, teal `#149E96`, gold `#E8AF35`.
Use the plate icon at 32 pixels or larger and the full wordmark at 240 pixels
or larger. Leave at least half a well's diameter clear around the design.

Regenerate the assets with:

```bash
python scripts/generate_logo.py
```

The generator uses Matplotlib and DejaVu Sans Bold. [preview.png](preview.png)
shows the primary design on a light background.
