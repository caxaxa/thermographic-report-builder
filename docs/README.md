# Thermographic Report Builder Documentation

This directory contains comprehensive technical documentation for the thermographic report builder system.

## Documentation Index

### Core Features

- **[Flight Appendix Documentation](FLIGHT_APPENDIX.md)** - Complete guide to the automated flight statistics and visualization appendix
  - Data sources and coordinate transformations
  - GPS prediction error metrics (critical quality indicator)
  - Flight statistics calculations
  - Visualization generation
  - Troubleshooting guide

- **[Thermal Backtracking Architecture](THERMAL_BACKTRACKING_ARCHITECTURE.md)** - Deep dive into source-map backtracking system
  - How defects are mapped from orthophoto to raw thermal images
  - Coordinate transformation chain
  - Camera projection and reprojection
  - Fallback strategies

### Guides

- **[Troubleshooting Guide](TROUBLESHOOTING.md)** - Common issues and solutions
  - Error diagnosis
  - Log analysis
  - Recovery procedures

### Planning Documents

- **[Flight Visualization Implementation Plan](FLIGHT_VISUALIZATION_IMPLEMENTATION_PLAN.md)** - Historical planning document for flight visualization feature
  - Original design decisions
  - Implementation roadmap
  - Technical considerations

## Quick Start

### For Developers

1. **New to the project?** Start with the main [README.md](../README.md) for architecture overview
2. **Working on reports?** Read [Flight Appendix Documentation](FLIGHT_APPENDIX.md)
3. **Working on GPS matching?** Read [Thermal Backtracking Architecture](THERMAL_BACKTRACKING_ARCHITECTURE.md)
4. **Debugging issues?** Check [Troubleshooting Guide](TROUBLESHOOTING.md)

### For Users

1. **Understanding flight statistics** → [Flight Appendix - Metrics Calculated](FLIGHT_APPENDIX.md#metrics-calculated)
2. **GPS error interpretation** → [Flight Appendix - GPS Prediction Error](FLIGHT_APPENDIX.md#gps-prediction-error-critical-metric)
3. **Report issues** → [Troubleshooting Guide](TROUBLESHOOTING.md)

## Key Concepts

### Coordinate Systems

**Critical Understanding**: The system deals with multiple coordinate systems:

1. **WGS84** (lat/lon/alt) - Standard GPS coordinates in degrees
   - Used in reports and visualizations
   - Range: -90° to +90° latitude, -180° to +180° longitude

2. **ENU (East-North-Up)** - Local tangent plane in meters
   - Used internally by OpenSfM
   - Relative to a reference point
   - **Must be converted to WGS84** before display

3. **UTM** - Universal Transverse Mercator projection
   - Used for orthophoto GeoTIFFs
   - Measured in meters

**Important**: Never use OpenSfM reconstruction.json coordinates directly without converting from ENU to WGS84!

### GPS Prediction Error

The **GPS prediction error** is the 3D Euclidean distance between:
- Raw GPS position from drone EXIF
- Bundle-adjusted optimized position from photogrammetry

**Why it matters**:
- Indicates reconstruction quality
- Shows GPS drift or environmental issues
- Critical for validating deliverable quality

**Interpretation**:
- **< 50m**: Excellent
- **50-100m**: Good
- **100-200m**: Moderate (verify with ground control)
- **> 200m**: Poor (investigate GPS issues)

### Source-Map Backtracking

A patented technique for mapping pixels from orthophoto → raw images:

1. ODM generates a "source-map" GeoTIFF during texturing
2. Each orthophoto pixel stores: `[view_id, raw_x, raw_y]`
3. System uses this to find exact thermal image and pixel coordinates
4. Enables accurate temperature measurement on raw thermal data

See [Thermal Backtracking Architecture](THERMAL_BACKTRACKING_ARCHITECTURE.md) for details.

## Documentation Standards

All documentation in this directory follows these standards:

### File Structure
- Start with overview and purpose
- Include table of contents for long documents
- Use clear section hierarchy
- End with references and version history

### Code Examples
- Include complete, runnable examples
- Show expected output
- Document parameters and return values

### Troubleshooting Sections
- List symptoms first
- Provide diagnosis steps
- Give clear solutions with commands
- Link to related documentation

### Updates
- Update version history when making changes
- Cross-reference related documentation
- Keep examples up-to-date with code

## Contributing

When adding new documentation:

1. **Choose the right location**:
   - Core features → New file in `/docs`
   - Code-specific docs → Docstrings in source files
   - Quick reference → Update main README.md

2. **Link from multiple places**:
   - Add to this index
   - Reference in main README.md
   - Link from related documentation

3. **Include examples**:
   - Show actual code usage
   - Provide sample inputs/outputs
   - Document edge cases

4. **Test accuracy**:
   - Verify commands actually work
   - Check file paths are correct
   - Validate code examples run

## Support

For questions or clarifications:

1. Check existing documentation in this directory
2. Review CloudWatch logs for the specific job
3. Search codebase for related implementations
4. Contact development team with specific questions

## Related Resources

### External Documentation
- [OpenSfM Documentation](https://opensfm.org/docs/)
- [ODM Documentation](https://docs.opendronemap.org/)
- [GDAL Python API](https://gdal.org/python/)
- [PyLaTeX Documentation](https://jeltef.github.io/PyLaTeX/)

### Internal Code
- [Flight Viz Module](../src/thermographic_report_builder/flight_viz/)
- [Processing Module](../src/thermographic_report_builder/processing/)
- [Report Builder](../src/thermographic_report_builder/report/)
