print("This is utility function for Sc-HistoMorphics")
from tifffile import imread,imwrite,TiffFile
import dask.array as da
import numpy as np
import pandas as pd
import cellprofiler_core.pipeline
from stardist.models import StarDist2D
from csbdeep.utils import normalize
import os
import cellprofiler_core.utilities.java
import cellprofiler_core.preferences
from multiprocessing import Pool

# import pdb

import skimage
def init_dirs(results_dir: str):
    """
    Initialize output directory structure for sc-HistoMorphics.

    Returns
    -------
    tuple of str
        Paths to label, measurement, plot, QuPath, and registration dirs.
    """
    label_dir = os.path.join(results_dir, "Tissue")
    meas_dir = os.path.join(results_dir, "Measurements")
    plot_dir = os.path.join(results_dir, "Plots")
    
    for d in [label_dir, meas_dir, plot_dir]:
        os.makedirs(d, exist_ok=True)

    return label_dir, meas_dir, plot_dir

    
def segment_nuclei(he_image,
    label_path,
    prob_thresh= 0.3,
    nms_thresh=0.1,
    block_size=(1024, 1024, 3),
    min_overlap = 110,
    force=False) -> np.ndarray:
    """
    Segment nuclei from an H&E image using StarDist (2D_versatile_he).

    Large images are processed block-wise using `predict_instances_big`.
    If a segmentation mask already exists, it is loaded unless `force=True`.

    Parameters
    ----------
    he_image : np.ndarray
        RGB H&E image with shape (Y, X, 3).
    label_path : str
        Path to save or load the nuclei segmentation mask (TIFF).
    prob_thresh : float, optional
        StarDist object probability threshold.
    nms_thresh : float, optional
        Non-maximum suppression threshold.
    block_size : tuple of int, optional
        Block size for large-image inference (Y, X, C).
        Default is (1024, 1024, 3).
    min_overlap : int, optional
        Minimum overlap (in pixels) between adjacent blocks.
    force : bool, optional
        If True, recompute segmentation even if `label_path` exists.

    Returns
    -------
    np.ndarray
        Integer-labeled nuclei mask with background = 0.
    """
    if not os.path.isfile(label_path) or force:
        print("Segmenting nuclei using StarDist (H&E)...")

        model = StarDist2D.from_pretrained("2D_versatile_he")

        # Normalize image as required by StarDist
        img_norm = normalize(he_image)

        labels, _ = model.predict_instances_big(
            img_norm,
            axes="YXC",
            block_size=block_size,
            min_overlap=min_overlap,
            prob_thresh=prob_thresh,
            nms_thresh=nms_thresh,
        )

        print("Finished model prediction, saving segmentation mask.")
        imwrite(label_path, labels.astype(np.int32))

    else:
        print("Segmentation mask exists. Loading from disk.")
        labels = imread(label_path).astype(np.int32)

    # Ensure background = 0 and all labels are positive
    min_label = abs(labels.min()) + 1
    labels = labels + min_label
    labels[labels == min_label] = 0

    return labels

def get_tiff_resolution(filepath: str, verbose: bool = False) -> float:
    """
    Extract PhysicalSizeX (pixel size in microns) from OME-TIFF metadata.

    Parameters
    ----------
    filepath : str
        Path to OME-TIFF file.
    verbose : bool, optional
        If True, print the extracted resolution and unit.

    Returns
    -------
    float
        Physical pixel size in microns (µm).
    """
    with TiffFile(filepath) as tif:
        ome_metadata = tif.ome_metadata

    if ome_metadata is None:
        raise ValueError("No OME metadata found in TIFF file.")

    # Extract substring containing PhysicalSizeX
    start = ome_metadata.find('PhysicalSizeX="')
    if start == -1:
        raise ValueError("PhysicalSizeX not found in OME metadata.")

    start += len('PhysicalSizeX="')
    end = ome_metadata.find('"', start)

    resolution = float(ome_metadata[start:end])

    if verbose:
        print(f"PhysicalSizeX = {resolution} µm")

    return resolution
def objects2df(measurements, objects_name):
    """
    Convert CellProfiler object measurements into a NumPy object array.

    Parameters
    ----------
    measurements : cellprofiler_core.measurement.Measurements
        CellProfiler measurements object.
    objects_name : str
        Name of the object set (e.g. 'Cell', 'FilteredNuclei').

    Returns
    -------
    np.ndarray
        Object array with keys:
        - 'data' : (n_objects, n_features) float array
        - 'columns' : list of feature names
    """
    # Retrieve feature names
    features = list(measurements.get_feature_names(objects_name))
    if len(features) == 0:
        raise ValueError(f"No features found for object '{objects_name}'")

    # Number of objects inferred from the first feature
    first_measurement = measurements.get_measurement(objects_name, features[0])
    n_objects = first_measurement.shape[0]
    n_features = len(features)

    # Preallocate data matrix
    data = np.empty((n_objects, n_features), dtype=float)

    # Populate feature matrix
    for idx, feature in enumerate(features):
        values = measurements.get_measurement(objects_name, feature)

        # Defensive check for unexpected shapes
        if values.shape[0] != n_objects:
            raise ValueError(
                f"Feature '{feature}' has {values.shape[0]} objects; "
                f"expected {n_objects}"
            )

        data[:, idx] = values

    # Return structure consistent with original implementation
    return np.array(
        [
            ("data", data),
            ("columns", features),
        ],
        dtype=object,
    )


def objects_images_2df(measurements, objects_name, workspace):
    """
    Convert CellProfiler object measurements and associated images
    into a NumPy object array.

    This function extracts:
    - Object-level measurement matrix
    - Feature (column) names
    - Hematoxylin, Eosin, and cytoplasm images from the workspace

    Parameters
    ----------
    measurements : cellprofiler_core.measurement.Measurements
        CellProfiler measurements object.
    objects_name : str
        Name of the object set (e.g. 'FilteredNuclei').
    workspace : cellprofiler_core.workspace.Workspace
        Active CellProfiler workspace containing image set.

    Returns
    -------
    np.ndarray
        Object array with keys:
        - 'data' : (n_objects, n_features) float array
        - 'columns' : list of feature names
        - 'Hematoxylin' : 2D image
        - 'Eosin' : 2D image
        - 'cyto' : 3-channel cytoplasm image
    """
    # ------------------------------------------------------------------
    # Extract object-level measurements
    # ------------------------------------------------------------------
    features = list(measurements.get_feature_names(objects_name))
    if not features:
        raise ValueError(f"No features found for object '{objects_name}'")

    first = measurements.get_measurement(objects_name, features[0])
    n_objects = first.shape[0]
    n_features = len(features)

    data = np.empty((n_objects, n_features), dtype=float)

    for idx, feature in enumerate(features):
        values = measurements.get_measurement(objects_name, feature)
        if values.shape[0] != n_objects:
            raise ValueError(
                f"Feature '{feature}' has {values.shape[0]} objects; "
                f"expected {n_objects}"
            )
        data[:, idx] = values

    # ------------------------------------------------------------------
    # Extract images from the workspace
    # ------------------------------------------------------------------
    image_set = workspace.image_set

    try:
        hematoxylin = image_set.get_image("Hematoxylin").pixel_data
        eosin = image_set.get_image("Eosin").pixel_data
        cyto = image_set.get_image("CellOverlay").pixel_data
    except Exception as e:
        raise KeyError(
            "Required images not found in workspace image set. "
            "Expected: Hematoxylin, Eosin, CellOverlay"
        ) from e

    # ------------------------------------------------------------------
    # Return structure (unchanged contract)
    # ------------------------------------------------------------------
    return np.array(
        [
            ("data", data),
            ("columns", features),
            ("Hematoxylin", hematoxylin),
            ("Eosin", eosin),
            ("cyto", cyto),
        ],
        dtype=object,
    )

def compute_cp_dask(
    img: np.ndarray,
    labels: np.ndarray,
    chunks=(1024, 1024, 4),
):
    """
    Run CellProfiler feature extraction in Dask chunks.

    Returns
    -------
    featu : np.ndarray
        Raw CellProfiler outputs per chunk.
    skip_last_chunk : tuple(bool, bool)
        Indicates if boundary chunks were truncated.
    """
    dask_img = da.from_array(
        np.concatenate([img, labels[..., None]], axis=2),
        chunks=chunks,
    )

    features = dask_img.map_overlap(
        run_cp_on_chunk,
        drop_axis=2,
        dtype=object,
        chunks=(2, 2),####
        meta=np.empty((0), dtype=object),###
        depth={0: 30, 1: 30},
        boundary={0: 'periodic', 1: 'periodic'},
        trim=False,
    ).compute()

    skip_last_chunk = (
        dask_img.chunks[0][-1] < 30,
        dask_img.chunks[1][-1] < 30,
    )

    return features, skip_last_chunk
def run_cp_on_chunk(chunk):
    """
    Run a CellProfiler pipeline on a single chunk.
    """
    cellprofiler_core.preferences.set_headless()
    #cellprofiler_core.utilities.java.start_java()

    pipeline = cellprofiler_core.pipeline.Pipeline()
    pipeline.load(
        "/projects/activities/deepspace/team/elaheh/Scripts/CellProfiler/Cell_Quantification_SC2200165_EuE-31-Final2.cppipe"
    )

    img_rgb = chunk[..., :3].astype(np.uint8)
    label_img = chunk[..., 3].astype(np.int32)

    # Normalize labels
    min_label = abs(label_img.min()) + 1
    label_img += min_label
    label_img[label_img == min_label] = 0

    image_set_list = cellprofiler_core.image.ImageSetList()
    image_set = image_set_list.get_image_set(0)
    image_set.add("HnE", cellprofiler_core.image.Image(img_rgb, convert=True))

    object_set = cellprofiler_core.object.ObjectSet()
    objects = cellprofiler_core.object.Objects()
    objects.segmented = label_img
    object_set.add_objects(objects, "OrgNuc")

    measurements = cellprofiler_core.measurement.Measurements()

    # Run pipeline (skip input modules)
    for module in pipeline.modules()[4:]:
        workspace = cellprofiler_core.workspace.Workspace(
            pipeline=pipeline,
            module=module,
            image_set=image_set,
            image_set_list=image_set_list,
            object_set=object_set,
            measurements=measurements,
        )
        module.run(workspace)

    nuc_df = objects_images_2df(workspace.measurements, "FilteredNuclei",workspace)
    cell_df = objects2df(workspace.measurements, "Cell")

    # Merge features
    merged = nuc_df.copy()
    merged[0, 1] = np.concatenate([nuc_df[0, 1], cell_df[0, 1]], axis=1)
    merged[1, 1] = np.concatenate([nuc_df[1, 1], cell_df[1, 1]])

    cellprofiler_core.utilities.java.stop_java()
    return merged
def selected_features():

    """
    Return selected morphology and texture feature names.
    """
    Names=['AreaShape_Nuc_Center_col', 'AreaShape_Nuc_Center_row', 'AreaShape_Area', 'AR','AreaShape_BoundingBoxArea', 'AreaShape_CentralMoment_0_0', 'AreaShape_CentralMoment_0_1','AreaShape_CentralMoment_0_2', 'AreaShape_CentralMoment_0_3','AreaShape_CentralMoment_1_0', 'AreaShape_CentralMoment_1_1','AreaShape_CentralMoment_1_2', 'AreaShape_CentralMoment_1_3','AreaShape_CentralMoment_2_0', 'AreaShape_CentralMoment_2_1',
       'AreaShape_CentralMoment_2_2', 'AreaShape_CentralMoment_2_3','AreaShape_Compactness', 'AreaShape_ConvexArea','AreaShape_Eccentricity', 'AreaShape_EquivalentDiameter','AreaShape_Extent', 'AreaShape_FormFactor',
        'AreaShape_HuMoment_0', 'AreaShape_HuMoment_1', 'AreaShape_HuMoment_2','AreaShape_HuMoment_3', 'AreaShape_HuMoment_4', 'AreaShape_HuMoment_5','AreaShape_HuMoment_6', 'AreaShape_InertiaTensorEigenvalues_0', 'AreaShape_InertiaTensorEigenvalues_1', 'AreaShape_InertiaTensor_0_0','AreaShape_InertiaTensor_0_1', 'AreaShape_InertiaTensor_1_0','AreaShape_InertiaTensor_1_1', 'AreaShape_MajorAxisLength','AreaShape_MaxFeretDiameter', 'AreaShape_MaximumRadius','AreaShape_MeanRadius', 'AreaShape_MedianRadius','AreaShape_MinFeretDiameter', 'AreaShape_MinorAxisLength','AreaShape_Orientation','AreaShape_NormalizedMoment_0_2', 'AreaShape_NormalizedMoment_0_3',
       'AreaShape_NormalizedMoment_1_1','AreaShape_NormalizedMoment_1_2', 'AreaShape_NormalizedMoment_1_3','AreaShape_NormalizedMoment_2_0', 'AreaShape_NormalizedMoment_2_1','AreaShape_NormalizedMoment_2_2', 'AreaShape_NormalizedMoment_2_3','AreaShape_NormalizedMoment_3_0', 'AreaShape_NormalizedMoment_3_1','AreaShape_NormalizedMoment_3_2', 'AreaShape_NormalizedMoment_3_3','AreaShape_Perimeter', 'AreaShape_Solidity',
       'AreaShape_SpatialMoment_0_0', 'AreaShape_SpatialMoment_0_1','AreaShape_SpatialMoment_0_2', 'AreaShape_SpatialMoment_0_3','AreaShape_SpatialMoment_1_0', 'AreaShape_SpatialMoment_1_1','AreaShape_SpatialMoment_1_2', 'AreaShape_SpatialMoment_1_3','AreaShape_SpatialMoment_2_0', 'AreaShape_SpatialMoment_2_1','AreaShape_SpatialMoment_2_2', 'AreaShape_SpatialMoment_2_3','AreaShape_Zernike_0_0', 'AreaShape_Zernike_1_1',
       'AreaShape_Zernike_2_0', 'AreaShape_Zernike_2_2','AreaShape_Zernike_3_1', 'AreaShape_Zernike_3_3','AreaShape_Zernike_4_0', 'AreaShape_Zernike_4_2','AreaShape_Zernike_4_4', 'AreaShape_Zernike_5_1','AreaShape_Zernike_5_3', 'AreaShape_Zernike_5_5', 'AreaShape_Zernike_6_0', 'AreaShape_Zernike_6_2',
       'AreaShape_Zernike_6_4', 'AreaShape_Zernike_6_6',
       'AreaShape_Zernike_7_1', 'AreaShape_Zernike_7_3',
       'AreaShape_Zernike_7_5', 'AreaShape_Zernike_7_7',
      'AreaShape_Zernike_8_0', 'AreaShape_Zernike_8_2',
       'AreaShape_Zernike_8_4', 'AreaShape_Zernike_8_6',
       'AreaShape_Zernike_8_8', 'AreaShape_Zernike_9_1',
       'AreaShape_Zernike_9_3', 'AreaShape_Zernike_9_5',
       'AreaShape_Zernike_9_7', 'AreaShape_Zernike_9_9', 
       'Intensity_IntegratedIntensityEdge_Eosin',
       'Intensity_IntegratedIntensityEdge_Hematoxylin',
       'Intensity_IntegratedIntensity_Eosin',
       'Intensity_IntegratedIntensity_Hematoxylin',
       'Intensity_LowerQuartileIntensity_Eosin',
       'Intensity_LowerQuartileIntensity_Hematoxylin',
       'Intensity_MADIntensity_Eosin', 'Intensity_MADIntensity_Hematoxylin',
       'Intensity_MassDisplacement_Eosin',
       'Intensity_MassDisplacement_Hematoxylin',
       'Intensity_MaxIntensityEdge_Eosin',
       'Intensity_MaxIntensityEdge_Hematoxylin',
       'Intensity_MaxIntensity_Eosin', 'Intensity_MaxIntensity_Hematoxylin',
       'Intensity_MeanIntensityEdge_Eosin',
       'Intensity_MeanIntensityEdge_Hematoxylin',
       'Intensity_MeanIntensity_Eosin', 'Intensity_MeanIntensity_Hematoxylin',
       'Intensity_MedianIntensity_Eosin',
       'Intensity_MedianIntensity_Hematoxylin',
       'Intensity_MinIntensityEdge_Eosin',
       'Intensity_MinIntensityEdge_Hematoxylin',
       'Intensity_MinIntensity_Eosin', 'Intensity_MinIntensity_Hematoxylin',
       'Intensity_StdIntensityEdge_Eosin',
       'Intensity_StdIntensityEdge_Hematoxylin',
       'Intensity_StdIntensity_Eosin', 'Intensity_StdIntensity_Hematoxylin',
       'Intensity_UpperQuartileIntensity_Eosin',
       'Intensity_UpperQuartileIntensity_Hematoxylin',
       'Location_CenterMassIntensity_X_Eosin',
       'Location_CenterMassIntensity_X_Hematoxylin',
       'Location_CenterMassIntensity_Y_Eosin',
       'Location_CenterMassIntensity_Y_Hematoxylin',
       'Location_MaxIntensity_X_Eosin', 'Location_MaxIntensity_X_Hematoxylin',
       'Location_MaxIntensity_Y_Eosin', 'Location_MaxIntensity_Y_Hematoxylin',
       'RadialDistribution_FracAtD_Eosin_1of4',
       'RadialDistribution_FracAtD_Eosin_2of4',
       'RadialDistribution_FracAtD_Eosin_3of4',
       'RadialDistribution_FracAtD_Eosin_4of4',
       'RadialDistribution_FracAtD_Hematoxylin_1of4',
       'RadialDistribution_FracAtD_Hematoxylin_2of4',
       'RadialDistribution_FracAtD_Hematoxylin_3of4',
       'RadialDistribution_FracAtD_Hematoxylin_4of4',
       'RadialDistribution_MeanFrac_Eosin_1of4',
       'RadialDistribution_MeanFrac_Eosin_2of4',
       'RadialDistribution_MeanFrac_Eosin_3of4',
       'RadialDistribution_MeanFrac_Eosin_4of4',
       'RadialDistribution_MeanFrac_Hematoxylin_1of4',
       'RadialDistribution_MeanFrac_Hematoxylin_2of4',
       'RadialDistribution_MeanFrac_Hematoxylin_3of4',
       'RadialDistribution_MeanFrac_Hematoxylin_4of4',
       'RadialDistribution_RadialCV_Eosin_1of4',
       'RadialDistribution_RadialCV_Eosin_2of4',
       'RadialDistribution_RadialCV_Eosin_3of4',
       'RadialDistribution_RadialCV_Eosin_4of4',
       'RadialDistribution_RadialCV_Hematoxylin_1of4',
       'RadialDistribution_RadialCV_Hematoxylin_2of4',
       'RadialDistribution_RadialCV_Hematoxylin_3of4',
       'RadialDistribution_RadialCV_Hematoxylin_4of4',
       'Neighbors_AngleBetweenNeighbors_Adjacent',
       'Neighbors_FirstClosestDistance_Adjacent',
       'Neighbors_FirstClosestObjectNumber_Adjacent',
       'Neighbors_NumberOfNeighbors_Adjacent',
       'Neighbors_PercentTouching_Adjacent',
       'Neighbors_SecondClosestDistance_Adjacent',
       'Neighbors_SecondClosestObjectNumber_Adjacent']#,
       # 'equivalent_diameter_area',
       # 'perimeter_crofton']       
    
    
    texture=['Texture_AngularSecondMoment_Eosin_3_00_256',
       'Texture_AngularSecondMoment_Eosin_3_01_256',
       'Texture_AngularSecondMoment_Eosin_3_02_256',
       'Texture_AngularSecondMoment_Eosin_3_03_256',
       'Texture_AngularSecondMoment_Hematoxylin_3_00_256',
       'Texture_AngularSecondMoment_Hematoxylin_3_01_256',
       'Texture_AngularSecondMoment_Hematoxylin_3_02_256',
       'Texture_AngularSecondMoment_Hematoxylin_3_03_256',
       'Texture_Contrast_Eosin_3_00_256', 'Texture_Contrast_Eosin_3_01_256',
       'Texture_Contrast_Eosin_3_02_256', 'Texture_Contrast_Eosin_3_03_256',
       'Texture_Contrast_Hematoxylin_3_00_256',
       'Texture_Contrast_Hematoxylin_3_01_256',
       'Texture_Contrast_Hematoxylin_3_02_256',
       'Texture_Contrast_Hematoxylin_3_03_256',
       'Texture_Correlation_Eosin_3_00_256',
       'Texture_Correlation_Eosin_3_01_256','Texture_Correlation_Eosin_3_02_256',
       'Texture_Correlation_Eosin_3_03_256',
       'Texture_Correlation_Hematoxylin_3_00_256',
       'Texture_Correlation_Hematoxylin_3_01_256',
       'Texture_Correlation_Hematoxylin_3_02_256',
       'Texture_Correlation_Hematoxylin_3_03_256',
       'Texture_DifferenceEntropy_Eosin_3_00_256',
       'Texture_DifferenceEntropy_Eosin_3_01_256',
       'Texture_DifferenceEntropy_Eosin_3_02_256',
       'Texture_DifferenceEntropy_Eosin_3_03_256',
       'Texture_DifferenceEntropy_Hematoxylin_3_00_256',
       'Texture_DifferenceEntropy_Hematoxylin_3_01_256',
       'Texture_DifferenceEntropy_Hematoxylin_3_02_256',
       'Texture_DifferenceEntropy_Hematoxylin_3_03_256',
       'Texture_DifferenceVariance_Eosin_3_00_256',
       'Texture_DifferenceVariance_Eosin_3_01_256',
       'Texture_DifferenceVariance_Eosin_3_02_256',
       'Texture_DifferenceVariance_Eosin_3_03_256',
       'Texture_DifferenceVariance_Hematoxylin_3_00_256',
       'Texture_DifferenceVariance_Hematoxylin_3_01_256',
       'Texture_DifferenceVariance_Hematoxylin_3_02_256',
       'Texture_DifferenceVariance_Hematoxylin_3_03_256',
       'Texture_Entropy_Eosin_3_00_256', 'Texture_Entropy_Eosin_3_01_256',
       'Texture_Entropy_Eosin_3_02_256', 'Texture_Entropy_Eosin_3_03_256',
       'Texture_Entropy_Hematoxylin_3_00_256',
       'Texture_Entropy_Hematoxylin_3_01_256',
       'Texture_Entropy_Hematoxylin_3_02_256',
       'Texture_Entropy_Hematoxylin_3_03_256',
       'Texture_InfoMeas1_Eosin_3_00_256', 'Texture_InfoMeas1_Eosin_3_01_256',
       'Texture_InfoMeas1_Eosin_3_02_256', 'Texture_InfoMeas1_Eosin_3_03_256',
       'Texture_InfoMeas1_Hematoxylin_3_00_256',
       'Texture_InfoMeas1_Hematoxylin_3_01_256',
       'Texture_InfoMeas1_Hematoxylin_3_02_256',
       'Texture_InfoMeas1_Hematoxylin_3_03_256',
       'Texture_InfoMeas2_Eosin_3_00_256', 'Texture_InfoMeas2_Eosin_3_01_256',
       'Texture_InfoMeas2_Eosin_3_02_256', 'Texture_InfoMeas2_Eosin_3_03_256',
       'Texture_InfoMeas2_Hematoxylin_3_00_256',
       'Texture_InfoMeas2_Hematoxylin_3_01_256',
       'Texture_InfoMeas2_Hematoxylin_3_02_256',
       'Texture_InfoMeas2_Hematoxylin_3_03_256',
       'Texture_InverseDifferenceMoment_Eosin_3_00_256',
       'Texture_InverseDifferenceMoment_Eosin_3_01_256',
       'Texture_InverseDifferenceMoment_Eosin_3_02_256',
       'Texture_InverseDifferenceMoment_Eosin_3_03_256',
       'Texture_InverseDifferenceMoment_Hematoxylin_3_00_256',
       'Texture_InverseDifferenceMoment_Hematoxylin_3_01_256',
       'Texture_InverseDifferenceMoment_Hematoxylin_3_02_256',
       'Texture_InverseDifferenceMoment_Hematoxylin_3_03_256',
       'Texture_SumAverage_Eosin_3_00_256',
       'Texture_SumAverage_Eosin_3_01_256',
       'Texture_SumAverage_Eosin_3_02_256',
       'Texture_SumAverage_Eosin_3_03_256',
       'Texture_SumAverage_Hematoxylin_3_00_256',
       'Texture_SumAverage_Hematoxylin_3_01_256',
       'Texture_SumAverage_Hematoxylin_3_02_256',
       'Texture_SumAverage_Hematoxylin_3_03_256',
       'Texture_SumEntropy_Eosin_3_00_256',
       'Texture_SumEntropy_Eosin_3_01_256',
       'Texture_SumEntropy_Eosin_3_02_256',
       'Texture_SumEntropy_Eosin_3_03_256',
       'Texture_SumEntropy_Hematoxylin_3_00_256',
       'Texture_SumEntropy_Hematoxylin_3_01_256',
       'Texture_SumEntropy_Hematoxylin_3_02_256',
       'Texture_SumEntropy_Hematoxylin_3_03_256',
       'Texture_SumVariance_Eosin_3_00_256',
       'Texture_SumVariance_Eosin_3_01_256',
       'Texture_SumVariance_Eosin_3_02_256',
       'Texture_SumVariance_Eosin_3_03_256',
       'Texture_SumVariance_Hematoxylin_3_00_256',
       'Texture_SumVariance_Hematoxylin_3_01_256',
       'Texture_SumVariance_Hematoxylin_3_02_256',
       'Texture_SumVariance_Hematoxylin_3_03_256',
       'Texture_Variance_Eosin_3_00_256', 'Texture_Variance_Eosin_3_01_256',
       'Texture_Variance_Eosin_3_02_256', 'Texture_Variance_Eosin_3_03_256',
       'Texture_Variance_Hematoxylin_3_00_256',
       'Texture_Variance_Hematoxylin_3_01_256',
       'Texture_Variance_Hematoxylin_3_02_256',
       'Texture_Variance_Hematoxylin_3_03_256']
    
    return(Names,texture)
def clean_up_morphs_no_st_data(
    featu,
    output_prefix: str,
    chunks: int,
    shape,
    tag=""
):
    """
    Aggregate chunk-wise CellProfiler outputs into a single-cell table
    and export cytoplasm / H&E channel images.
    """
    names, texture = selected_features()

    All_matrix = np.empty(shape=(1,len(featu[1,1])))
    Hemotoxylin=np.zeros(shape=shape)
    cyto=np.zeros(shape=np.concatenate([shape,[3]]))
    Eosin=np.zeros(shape=shape)

    x_idx = list(featu[1, 1]).index("AreaShape_Center_X")
    y_idx = list(featu[1, 1]).index("AreaShape_Center_Y")

    layers = 5
    for r in range(featu.shape[0] // layers):
        for c in range(featu.shape[1] // 2):
            matrix=featu[r*layers,c*2+1]
            Hemo=featu[r*layers+2,c*2+1]
            Eo=featu[r*layers+3,c*2+1]
            cyo=featu[r*layers+4,c*2+1]
            
            if isinstance(matrix, float):
                continue

            matrix[:,x_idx]=matrix[:,x_idx]+(chunks)*c-30
            matrix[:,y_idx]=matrix[:,y_idx]+(chunks)*r-30


            All_matrix=np.concatenate([All_matrix,matrix])
                

            Hemotoxylin[(chunks)*r:(chunks)*(r+1),(chunks)*c:(chunks)*(c+1)]=Hemo[30:Hemo.shape[0]-30,30:Hemo.shape[1]-30]
            Eosin[(chunks)*r:(chunks)*(r+1),(chunks)*c:(chunks)*(c+1)]=Eo[30:Eo.shape[0]-30,30:Eo.shape[1]-30]
            cyto[(chunks)*r:(chunks)*(r+1),(chunks)*c:(chunks)*(c+1)]=cyo[30:cyo.shape[0]-30,30:cyo.shape[1]-30]

            
    
    # ---------------------------------------------------------------------
    # Build morphology DataFrame
    # ---------------------------------------------------------------------
    df_g = pd.DataFrame(All_matrix[1:,:], columns=featu[1, 1])
    
    # Round nucleus center coordinates to pixel resolution
    df_g["AreaShape_Center_X"] = df_g["AreaShape_Center_X"].round()
    df_g["AreaShape_Center_Y"] = df_g["AreaShape_Center_Y"].round()
    
    # Sort so the largest nucleus is kept per (X, Y) location
    df_g = df_g.sort_values(
        ["AreaShape_Center_X", "AreaShape_Center_Y", "AreaShape_Area"],
        ascending=[False, False, False],
    )
    
    # Drop duplicate nuclei with identical centers
    All = (
        df_g
        .drop_duplicates(subset=["AreaShape_Center_X", "AreaShape_Center_Y"])
        .reset_index(drop=True)
    )
    
    # Rename centers for downstream compatibility
    All["AreaShape_Nuc_Center_col"] = All["AreaShape_Center_X"]
    All["AreaShape_Nuc_Center_row"] = All["AreaShape_Center_Y"]
    # Aspect ratio (derived feature)
    All["AR"] = (
        All["AreaShape_MajorAxisLength"]
        / All["AreaShape_MinorAxisLength"]
    )
    # ---------------------------------------------------------------------
    # Rotation-invariant texture aggregation
    # ---------------------------------------------------------------------
    
    # Identify rotation-invariant feature prefixes
    rot_prefixes = (
        pd.Series(texture)
        .str.split("_")
        .str[:3]
        .str.join("_")
        .unique()
    )
    
    # Average rotation variants
    for prefix in rot_prefixes:
        cols = All.filter(like=prefix).columns
        if len(cols) > 0:
            All[prefix] = All[cols].mean(axis=1)
        
    
    # ---------------------------------------------------------------------
    # Feature selection and spatial filtering
    # ---------------------------------------------------------------------
    # Select core features + rotation-collapsed texture features
    number=len(names + sorted(set(rot_prefixes)) + ["AR"])
    print(f"Length of features is : {number}")
    All2 = All.loc[:, names + sorted(set(rot_prefixes))].copy()
    
    # Remove nuclei close to image borders (artifact-prone regions)
    margin = 30
    h, w = shape
    
    spatial_mask = (
        (All2["AreaShape_Nuc_Center_col"] > margin) &
        (All2["AreaShape_Nuc_Center_col"] < w - margin) &
        (All2["AreaShape_Nuc_Center_row"] > margin) &
        (All2["AreaShape_Nuc_Center_row"] < h - margin)
    )
    
    All2 = All2.loc[spatial_mask].copy()

    
    
    

    
    All.to_csv(output_prefix + "_Measurements"+tag+".csv", index=False)

    imwrite(output_prefix +tag+ "_hemotoxylin.tif", skimage.util.img_as_int(Hemotoxylin))   
    imwrite(output_prefix+tag+"_cytoplasm.tif", skimage.util.img_as_int(cyto))
    imwrite(output_prefix+tag+"_eosin.tif", skimage.util.img_as_int(Eosin))
    
    return All2, Hemotoxylin, Eosin, cyto

    
def compute_or_load_morphology(
    he_image: np.ndarray,
    labels: np.ndarray,
    res_dir: str,
    chunk_size: int = 4000,
    force=False,
    tag=""
):
    """
    Compute CellProfiler morphology features if they do not exist.

    -------clean_up_morphs_no_st_data
    Morph : pandas.DataFrame
        Morphology feature table.
    Hematoxylin, Eosin, Cytoplasm : np.ndarray
        Derived H&E channels and cytoplasm mask.
    """
    if not os.path.isfile(f"{res_dir}_Measurements{tag}.csv") or force:
        print("CellProfiler measurements not found. Computing...")

        features, _ =compute_cp_dask(
            he_image, labels, chunks=(chunk_size, chunk_size, 4)
        )

        Morph, Hematoxylin, Eosin, Cytoplasm = (
            clean_up_morphs_no_st_data(
                features,
                res_dir,
                chunks=chunk_size,
                shape=he_image.shape[:2],
            )
        )
        print(f" Morph features shape is : {Morph.shape}")
    else:
        print("Loading existing CellProfiler measurements...")
        Morph = pd.read_csv(f"{res_dir}_Measurements{tag}.csv")
        Hematoxylin = imread(f"{res_dir}{tag}_hemotoxylin.tif")
        Eosin = imread(f"{res_dir}{tag}_eosin.tif")
        Cytoplasm = imread(f"{res_dir}{tag}_cytoplasm.tif")
    return Morph, Hematoxylin, Eosin, Cytoplasm


def load_xenium_outputs(in_dir: str):
    """Load Xenium nuclei centers, boundaries, and transformation matrix."""
    nuc_center = pd.read_parquet(f"{in_dir}/outs/cells.parquet")
    nuc_bound = pd.read_parquet(f"{in_dir}/outs/nucleus_boundaries.parquet")

    transform = np.loadtxt(
        f"{in_dir}/Xenium_FFPE_Human_Breast_Cancer_Rep1_he_imagealignment.csv",
        delimiter=",",
    )

    
    return nuc_center, nuc_bound, transform


def transform_coords(df, x_col, y_col, transform, res_um):
    """
    Transform coordinates from Xenium DAPI space into H&E image space.
    """
    M = transform[:2, :2]
    Tr = transform[:2, 2]
    coords = df[[x_col, y_col]].to_numpy() / res_um
    coords = coords - Tr
    coords = coords @ np.linalg.inv(M.T)

    out = pd.DataFrame(coords, columns=[x_col, y_col])
    out["cell_id"] = df["cell_id"].values
    return out


_XE_X = _XE_Y = _XE_ID = None


def _init_pool(xe_x, xe_y, xe_id):
    global _XE_X, _XE_Y, _XE_ID
    _XE_X = xe_x
    _XE_Y = xe_y
    _XE_ID = xe_id


def _match_one(args):
    """Match a morphology nucleus to the closest Xenium cell within radius."""
    x, y, r = args
    mask = (
        (_XE_X > x - r)
        & (_XE_X < x + r)
        & (_XE_Y > y - r)
        & (_XE_Y < y + r)
    )

    if not mask.any():
        return np.nan

    idx = np.flatnonzero(mask)
    dx = _XE_X[idx] - x
    dy = _XE_Y[idx] - y
    return _XE_ID[idx[np.argmin(dx**2 + dy**2)]]


def match_cells_parallel(
    morph: pd.DataFrame,
    nuc_center_hne: pd.DataFrame,
    n_jobs: int = 20,
    chunksize: int = 1000,
):
    """
    Match H&E morphology nuclei to Xenium cells using spatial proximity.
    """
    xe_x = nuc_center_hne["x_centroid"].to_numpy()
    xe_y = nuc_center_hne["y_centroid"].to_numpy()
    xe_id = nuc_center_hne["cell_id"].to_numpy()

    tasks = zip(
        morph["AreaShape_Nuc_Center_col"].to_numpy(),
        morph["AreaShape_Nuc_Center_row"].to_numpy(),
        morph["AreaShape_MinorAxisLength"].to_numpy() / 2,
    )

    #t0 = time.perf_counter()
    with Pool(
        n_jobs, initializer=_init_pool, initargs=(xe_x, xe_y, xe_id)
    ) as pool:
        matched = list(pool.imap(_match_one, tasks, chunksize))

    #elapsed = time.perf_counter() - t0

    out = morph[
        ["AreaShape_Nuc_Center_col", "AreaShape_Nuc_Center_row"]
    ].copy()
    out["cell_id"] = matched

    out["condition"] = np.where(
        out["cell_id"].isna(), "Morph_only", "Both"
    )

    # Identify multi-mapped Xenium cells
    counts = out["cell_id"].value_counts(dropna=True)
    multi_ids = counts[counts > 1].index

    mask = out["cell_id"].isin(multi_ids)
    out.loc[mask, "condition"] = "belongs_with_another_cellto_one_cell_ID"
    out.loc[mask, "cell_id"] = "Multiple"

    #summary = out.groupby("condition").size().reset_index(name="n")
    return out#, summary, elapsed

def slurm_logical_cores():
    # Prefer SLURM_* env if provided
    cpt = int(os.getenv("SLURM_CPUS_PER_TASK", "1"))
    ntasks = int(os.getenv("SLURM_NTASKS", "1"))
    total = max(1, cpt * ntasks)

    # Fallback to cpuset if available
    try:
        with open("/sys/fs/cgroup/cpuset/cpuset.cpus") as f:
            txt = f.read().strip()
        def count_list(spec):
            cnt = 0
            for part in spec.split(","):
                if "-" in part:
                    a, b = map(int, part.split("-"))
                    cnt += (b - a + 1)
                else:
                    cnt += 1
            return cnt
        cpus = count_list(txt) if txt else total
        total = max(1, min(total or cpus, cpus))
    except Exception:
        pass
    return total