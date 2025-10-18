# Docker Image Optimization Notes

## Current Image Size: ~1.2 GB

### Breakdown
- Base `python:3.11-slim`: ~130 MB
- LaTeX packages: ~650 MB
  - `texlive-latex-base`: ~200 MB (pdflatex, basic fonts)
  - `texlive-latex-extra`: ~400 MB (geometry, fancyhdr, subfig, tikz, xcolor, booktabs)
  - `texlive-lang-portuguese`: ~50 MB (babel-brazil)
- Python packages: ~300 MB (OpenCV, rasterio, numpy, boto3, etc.)
- Application code: ~5 MB

### LaTeX Packages Used

Our report generation requires these specific LaTeX packages:

```latex
\usepackage[utf8]{inputenc}       % UTF-8 support (in base)
\usepackage[brazil]{babel}        % Portuguese language (texlive-lang-portuguese)
\usepackage{graphicx}             % Include images (in base)
\usepackage{placeins}             % FloatBarrier (in latex-extra)
\usepackage{calc}                 % Calculations (in latex-extra)
\usepackage{tikz}                 % Vector graphics (in latex-extra)
\usepackage{xcolor}               % Colors (in latex-extra)
\usepackage{fancyhdr}             % Headers/footers (in latex-extra)
\usepackage{subfig}               % Subfigures (in latex-extra)
\usepackage{geometry}             % Page layout (in latex-extra)
\usepackage{booktabs}             % Professional tables (in latex-extra)
```

### Why We Can't Go Smaller

**Option 1: Install only specific .sty files** ❌
- Debian/Ubuntu doesn't provide granular texlive packages
- Each package has complex dependency trees
- Would need to manually download .sty files from CTAN (brittle)

**Option 2: Use Alpine Linux** ❌
- Alpine's texlive packages are even larger
- More compatibility issues with Python packages (musl vs glibc)
- Not worth the complexity

**Option 3: Remove LaTeX entirely** ⚠️
- Would require complete rewrite of report generation
- Alternatives: wkhtmltopdf, ReportLab (Python PDF), Typst
- Trade-off: LaTeX produces much higher quality PDFs

### Optimization Strategies (Future)

#### Short-term (Keep LaTeX)
1. ✅ **Multi-stage build** - Already implemented
2. ✅ **Remove unnecessary packages** - Already minimal
3. ✅ **Clean apt cache** - Already done
4. 🔄 **Layer caching** - Build LaTeX layer once, reuse

#### Long-term (Architectural Changes)
1. **Separate Report Service**
   - Deploy LaTeX container separately
   - Report builder calls API to generate PDF
   - Amortize LaTeX overhead across many requests

2. **Pre-rendered Templates**
   - Generate LaTeX templates at build time
   - Runtime only fills in data
   - Reduces compilation time, not image size

3. **Alternative PDF Generation**
   - **ReportLab** (Python): ~50 MB total image
   - **wkhtmltopdf**: ~300 MB total image
   - **Typst**: ~200 MB total image (modern LaTeX alternative)

   Trade-off: Need to rewrite all report templates

### Recommendation

**For now, keep LaTeX** (1.2 GB image):
- ✅ High-quality PDFs with professional typography
- ✅ Proven technology
- ✅ Legacy code compatibility
- ✅ Only downloaded once per Batch compute environment
- ✅ Acceptable for infrequent jobs (1-2 reports per hour)

**Future migration path**:
- If we need 100+ reports/hour, consider dedicated report service
- If storage costs become issue, consider ReportLab rewrite
- Monitor actual usage patterns first

### Build Optimization

To speed up builds, structure layers from least to most frequently changed:

```dockerfile
# ✅ Current structure (optimal)
1. Base image (cached)
2. System dependencies including LaTeX (cached until OS update)
3. Python dependencies (cached until pyproject.toml changes)
4. Application code (changes frequently, but small)
```

### ECR Storage Costs

With 1.2 GB image:
- **Storage**: $0.10/GB/month = ~$0.12/month
- **Transfer**: Minimal (stays in us-east-2)
- **Total cost**: Negligible (<$5/year)

**Conclusion**: Image size is acceptable for this use case.
