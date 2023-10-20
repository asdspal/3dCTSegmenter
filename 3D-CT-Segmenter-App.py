import io
from PIL import Image
import streamlit as st
from pathlib import Path
import cv2
import copy
from typing import Dict, Any
import time
import os
import sys
import zipfile
from os import PathLike
import urllib
sys.path.append("../utils")
import numpy as np
from monai.transforms import LoadImage
import openvino as ov

from custom_segmentation import SegmentationModel

models_dir = Path('pretrained_model')

ir_model_url = 'https://storage.openvinotoolkit.org/repositories/openvino_notebooks/models/kidney-segmentation-kits19/FP16-INT8/'
ir_model_name_xml = 'quantized_unet_kits19.xml'
ir_model_name_bin = 'quantized_unet_kits19.bin'
device = 'AUTO'

#download_file(ir_model_url + ir_model_name_xml, filename=ir_model_name_xml, directory=models_dir)
#download_file(ir_model_url + ir_model_name_bin, filename=ir_model_name_bin, directory=models_dir)

MODEL_PATH = models_dir / ir_model_name_xml
core = ov.Core()
segmentation_model = SegmentationModel(
    ie=core, model_path=Path(MODEL_PATH), sigmoid=True, rotate_and_flip=True
)


def load_case(CASE):
    

    
    if CASE is not None:
        BASEDIR = Path("kits19_frames_1")
        # The CT scan case number. For example: 16 for data from the case_00016 directory.
        # Currently only 117 is supported.


        case_path = BASEDIR / f"case_{CASE}"

        if not case_path.exists():
            
            url = f"https://storage.openvinotoolkit.org/data/test_data/openvino_notebooks/kits19/case_{CASE}.zip"
            filename = download_file(
                url
            )
            with zipfile.ZipFile(filename, "r") as zip_ref:
                zip_ref.extractall(path=BASEDIR)
            os.remove(filename)  # remove zipfile
            print(f"Downloaded and extracted data for case_{CASE}")
        else:
            print(f"Data for case_{CASE} exists")

        image_paths = sorted(case_path.glob("imaging_frames/*jpg"))
        
        framebuf = []

        next_frame_id = 0
        reader = LoadImage(image_only=True, dtype=np.uint8)

        while next_frame_id < len(image_paths) - 1:
            image_path = image_paths[next_frame_id]
            image = reader(str(image_path))
            framebuf.append(image)
            next_frame_id += 1

        return(framebuf)
    else:
        return(None)



def load_model():
    core = ov.Core()
    segmentation_model = SegmentationModel(
        ie=core, model_path=Path(MODEL_PATH), sigmoid=True, rotate_and_flip=True
    )
    return segmentation_model


# Define a callback function that runs every time the asynchronous pipeline completes inference on a frame
def completion_callback(infer_request: ov.InferRequest, user_data: Dict[str, Any],) -> None:
    preprocess_meta = user_data['preprocess_meta']
    
    raw_outputs = {out.any_name: copy.deepcopy(res.data) for out, res in zip(infer_request.model_outputs, infer_request.output_tensors)}
    frame = segmentation_model.postprocess(raw_outputs, preprocess_meta)

    _, encoded_img = cv2.imencode(".jpg", frame, params=[cv2.IMWRITE_JPEG_QUALITY, 90])
    # Create IPython image
    
    st.write(frame)
    st.image(frame)

    # Display the image in this notebook
    #display.clear_output(wait=True)
    #display.display(i)
    

def predict( segmentation_model, framebuf):

    
    compiled_model = core.compile_model(segmentation_model.net, device)

    
    imageLocation = st.empty()
    start_time = time.time()
    for i, input_frame in enumerate(framebuf):
        inputs, preprocessing_meta = segmentation_model.preprocess({segmentation_model.net.input(0): input_frame})
        raw_outputs = compiled_model(inputs)
        frame = segmentation_model.postprocess(raw_outputs, preprocessing_meta)
        imageLocation.image(frame)
        time.sleep(1)
        
    # Wait until all inference requests in the AsyncInferQueue are completed
    

def main():
    st.title('Pretrained model demo')
    CASE = st.selectbox( 'Select patient case for demo (case number of KiTs19 dataset)', [117, 2, 30, 16]
                            )
    print(CASE)
    if CASE is not None:
        CASE = str(CASE).zfill(5)
        print("case_Value", CASE)
        image = load_case(CASE)
        print(len(image))
        model = load_model()
        result = st.button('Run on case')
        if result:
            st.write('Calculating results...')
            predict(model, image)


if __name__ == '__main__':
    main()