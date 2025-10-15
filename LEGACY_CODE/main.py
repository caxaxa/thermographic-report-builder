
from image_processing.processing import generate_defect_map, annotate_and_downscale_orthophoto, process_and_rename_images, annotate_and_crop_defect_area 
from helpers.helpers import load_orthophoto , flatten_panel_defects_dict,sort_key
from report_builder.report_generator import generate_report
from report_builder.tex_to_pdf import run_pdflatex
from DXF_layers.layer_generator import process_geotiff
from DXF_layers.dxf_processing import dxf_file_path
import os
import json

if __name__ == "__main__":
    # Get the current directory of the script
    current_dir = os.path.dirname(os.path.abspath(__file__))

    #get farea name from user DB
    area_name = 'Area 1'

    # Construct the relative path
    ortho_path = os.path.join(current_dir, "Inputs", "manual_labeling", "map_cropped.tif")
    print('Loading Ortophoto')
    ortho_img = load_orthophoto(ortho_path)

    # Path for the dxf directory
    print('Searching for .dfx file')

    dxf_file_path = dxf_file_path(current_dir)

    # Create Annotation Counturns
    json_path = os.path.join(current_dir, "Inputs", "manual_labeling", "labels_cropped.json")

    report_dir = os.path.join(current_dir,'output', "report", "")

    report_images_dir = os.path.join(current_dir, report_dir, "report_images", "")

    new_ortho_path = os.path.join(current_dir, report_images_dir, "ortho.png")

    print('Annotating and Downscaling Ortophoto')
    #Load JSON data
    annotate_and_downscale_orthophoto(
        ortho_path = ortho_path, 
        json_path = json_path, 
        output_path = new_ortho_path, 
        scale_factor=0.25
    )

    print('Orthophoto Downscaled and Saved')

    layer_image_path = os.path.join(current_dir, report_images_dir, "layer_img.pdf")
                                 
    # Creating a deffects dict
    panel_defects, image_layer_vector = generate_defect_map(
        tif_path = ortho_path,
        annotation_json_path = json_path,
        alignment='vertical',
        output_image=layer_image_path
    )
                         

    raw_images_dir = os.path.join(current_dir, "Inputs", "raw_images","")

    print('Crop the map Images')

    annotate_and_crop_defect_area(
        tif_path = ortho_path,
        panel_defects_dict = panel_defects,
        image_layer_vector = image_layer_vector,  # or another reference layer
        default_panel_width=127,
        crop_panel_size=5,
        output_dir=report_images_dir,
        scale_factor = 0.5
    )

    print('Locate the Panel Original Drone Images')
    process_and_rename_images(
        raw_images_dir=raw_images_dir,
        output_dir=report_images_dir,
        panel_defects_dict= panel_defects,
        alignment='vertical',  # or 'horizontal', not strictly used inside this code
        quality = 70
    )

    print('Images successfully prepared')


    # Generating Latex Report:




    #stats_file = os.path.join(report_dir,  'report_images' ,'stats.json')
    flatned_panel_defects = flatten_panel_defects_dict(panel_defects) 

    sorted_defects_items = sorted(flatned_panel_defects.items(), key=sort_key)

    flatned_panel_defects = {k: v for k, v in sorted_defects_items}


    latex_code = generate_report(flatned_panel_defects, 'Área Teste', current_dir)

    latex_report_path = os.path.join(current_dir, report_dir, "report.tex")

    with open(latex_report_path, "w", encoding='utf-8') as f:
        f.write(latex_code)

    print('Report Latex File Created')

    #Compile the .tex file
    run_pdflatex("report.tex", report_dir)

    print('First Compilation Successful')

    # Compile second time... Latex may need extra compilation to run packeges and references properly
    run_pdflatex("report.tex", report_dir)

    print('Second Compilation Successful')

    print('PDF report successfully generated')

    # #Generate DFX Layers for the area

    # print('Saving Dataframe in XLS')

    # xlsx_path = os.path.join(current_dir, 'Output' , f'{area_name}.xlsx')

    # save_xls(df, xlsx_path)

    # print('XLS Successfully Created')

    # if dxf_file_path:
    #     print('Creating DXF Layers')
    #     # Instantiate the GeoImageProcessor with paths to the GeoTIFF and DXF files.
    #     process_geotiff(ortho_path, dxf_file_path,blue_mask, red_mask,area_name)
    #     print('DXF layers successfully generated')
    # else: print('No DXF Available')

    # print('GRETA Finished Successfully!')