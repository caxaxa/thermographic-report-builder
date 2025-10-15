# Documentation for Greta System Report Generator

Greta System report generator is a sophisticated application designed to pinpoint and report hotspots on solar panels mounted on trackers. By analyzing given orthophotos, and optionally, an AutoCAD DXF file, the application offers a comprehensive report on the identified hotspots and updates the DXF layers to show affected panels.

## Directory Structure
As observed from the provided image, the directory structure is as follows:

```
Greta_Tracker_LOC
│
├── DXF_layers
│   ├── layer_generator.py
│   ├── dxf_processing.py
│   └── ... (Other related modules)
│
├── helpers
│   ├── helpers.py
│   └── ... (Other helper modules)
│
├── image_processing
│   ├── processing.py
│   ├── image_creation.py
│   └── ... (Other image processing modules)
│
├── Inputs
│   ├── ... (Input files, e.g., odm_orthophoto.tif)
│
├── Output
│   ├── ... (Generated output files)
│
├── report
│   ├── report_images
│   │   └── ... (Processed images for the report)
│   └── ... (Generated report files, e.g., report.tex)
│
└── report_generation
    ├── report_generator.py
    ├── df_generator.py
    ├── tex_to_pdf.py
    ├── preparing_images.py
    └── ... (Other report generation modules)
```

## Overview of the Application Flow
1. **Load Orthophoto**: Using the `load_orthophoto` function from the `helpers` module, the application loads the orthophoto from the specified location.
2. **Create Masks**: Based on the loaded orthophoto, blue and red masks are created to help in the identification of the trackers and hotspots, respectively.
3. **Generate Affected Panels Coordinates**: By leveraging the blue and red masks, the application determines the coordinates of the affected solar panels.
4. **Generate Layer Image**: The system saves an image that showcases the trackers, the hotspots, and the numbered affected panels.
5. **Prepare Dataframe for Report**: The application converts the affected panels' coordinates into a structured dataframe format, then transforms these coordinates for report generation.
6. **Prepare Images for Report**: All relevant images for the report are located and copied to the appropriate directory.
7. **Generate LaTeX Report**: A LaTeX report is generated based on the processed data and saved to the report directory.
8. **Compile LaTeX to PDF**: The LaTeX report can be compiled into a PDF document using the `run_pdflatex` function, though this step appears to be commented out in the provided code.
9. **Generate DXF Layers**: If a DXF file is provided, the application will:
    - Detect the solar panel trackers using the blue mask.
    - Detect the hotspots using the red mask.
    - Draw and fill the hotspot contours on the DXF file.
    - Annotate the detected panels on the trackers.
    - Save the modified DXF file with the new annotations and layers.

## Dependencies
The application uses several third-party libraries, such as `cv2`, to facilitate image processing. Ensure that all required libraries, as mentioned in the `requirements.txt`, are installed before running the application.

## Execution
To execute the application, simply run the `main.py` file. Ensure all input files are present in the respective directories, and the directories have the necessary permissions for reading and writing.

## Further Information
Each module (e.g., `image_processing`, `DXF_layers`, `report_generation`) contains specific functions that contribute to the overall functionality of the application. For more detailed information about each module and its functions, it's recommended to check the module-specific documentation and comments in the source code.

---

This is a basic documentation of the Greta System report generator. For any advanced configurations, user guidance, or detailed explanations, further and more in-depth documentation might be required.