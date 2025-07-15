# Scikit-learn: Land Cover Classification Modelling By Machine Learning Algorithms

This project provides a comprehensive, end-to-end solution for land cover classification. It begins with the loading and preparation of remote sensing data (images from Landsat and PALSAR satellites), proceeds with accurate classification using a machine learning model, and culminates in the generation of a high-quality land cover map with geospatial referencing and embedded colormap. The classification results are output in a georeferenced image format (GeoTIFF) and integrate a predefined color coding scheme, aiming to provide accurate, timely, and easily visualized land cover information.

### Library
The library ```Scikit-learn``` provides a wide range of supervised and unsupervised learning algorithms, along with tools for **model fitting, data preprocessing, model selection, and evaluation**. It's built on NumPy, SciPy, and Matplotlib, making it highly efficient for numerical and scientific computing.
```python
import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import rasterio as rio
from rasterio.enums import Resampling
import json
import pandas as pd
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from PIL import ImageColor
import skimage as ski
from skimage.exposure import rescale_intensity
import scipy
from rasterio.features import shapes
```
### Data Preparation
Input data file paths.
```python
lc_param_loc = "data/lc.json"
sample_loc = "data/Sample_LC_v1.geojson"
palsar_loc = "data/Palsar_2025.tif"
landsat_loc = "data/Landsat_2025.tif"
```

### Land Cover Parameter Processing
This section acts as the data preparation and visualization backbone for the land cover classification. It takes raw land cover definitions and transforms them into a structured, normalized, and visually coherent format.

#### 1. Standardizing Land Cover Definitions
The code first loads land cover parameters from a JSON file (```lc = json.load(open(lc_dir))```). This external file likely contains a standardized definition of each land cover class, including their original values, names (labels), and associated color palettes.

#### 2. Normalizing and Mapping Values
The original land cover values might not be sequential or optimized for direct use in all processing steps. This section normalizes these values (```lc_df["values_normalize"] = lc_df.index + 1```) and creates several mapping dictionaries (```dict_values```, ```dict_label```, ```dict_palette```, ```dict_palette_hex```).

These dictionaries are vital for:
  * Converting original class IDs to a normalized, sequential range: This can simplify array indexing or further computations.
  * Associating normalized values with human-readable labels: ```dict_label``` allows for easy interpretation of numerical classification results.
  * Mapping class IDs to specific colors: ```dict_palette``` and ```dict_palette_hex``` are fundamental for generating visually meaningful maps.

#### 3. Creating a Custom Colormap for Visualization
A ```matplotlib.colors.ListedColormap``` (```cmap = ListedColormap(palette)```) is created directly from the defined color palette. This is highly important because it ensures that:
  * Each land cover class in the resulting classified image will be displayed with its **exact**, **predefined color**.
  * The visualization is consistent and immediately understandable, without requiring manual color assignments in external software.

#### 4. Generating a Visual Legend
The code also generates "patches" (```patches = [mpatches.Patch(color=palette[i], label=labels[i]) for i in range(len(values))]```) for a legend. This is essential for:
  * **Interpretability**: A legend clearly explains what each color on the map represents, making the land cover classification easily decipherable for any viewer.
  * **Professional Presentation**: It allows for the creation of publication-quality maps where all necessary information is provided within the visualization itself.

### Load The Sample Data
It is **critical for handling and visualizing the training data** in the land cover classification project. Its importance can be broken down into the following key aspects:

#### 1. Loading and Preparing Training Data
This part loads the geospatial training samples from a GeoJSON file (```sample_dir```) using ```geopandas```.
```python
# Load sample
sample = gpd.read_file(sample_dir)
sample["value"] = sample["lc"].map(dict_values)
sample["label"] = sample["value"].map(dict_label)
```

#### 2. Visualizing Training Samples
This segment visualizes the loaded training samples.
```python
# Plot sample
sample.plot(column="value", cmap=cmap, markersize=1)
plt.legend(**legend)
```

#### 3. Preparing for Feature Extraction (Sample with Extract)
Prepare the samples for the actual feature extraction from the satellite imagery.
```python
# Sample with extract
sample_extract = sample.copy()
coords = [
    (x, y) for x, y in zip(sample_extract["geometry"].x, sample_extract["geometry"].y)
]
print(sample_extract.shape)
```

### Load The Landsat data and Landsat Raster Values Prsentation By Extraction
The section is **fundamental for processing and utilizing Landsat imagery** in the land cover classification project. It handles the initial loading of the imagery, creates a visually useful composite, and most importantly, extracts the critical spectral information at the exact locations of training samples for model input.

#### 1. Loading and Initial Processing of Landsat Imagery
```python
# Load landsat image
landsat = rio.open(landsat_dir)
landsat_image = landsat.read() / 1e4
```

#### 2. Creating a False Color Composite for Visualization
Generates a false color composite (FCC) image and plots it. While not directly used for classification input, FCCs are incredibly important for visual interpretation in remote sensing.
  * ```rescale_intensity(...)```: This function adjusts the intensity (brightness and contrast) of individual bands (Bands 5, 6, and 7 of Landsat 8/9, commonly used for FCCs where near-infrared (NIR) is mapped to red, short-wave infrared 1 (SWIR1) to green, and short-wave infrared 2 (SWIR2) to blue). 
```python
# False color composite
out_range = (0, 1)
red = rescale_intensity(landsat_image[4], in_range=(0.1, 0.4), out_range=out_range)
green = rescale_intensity(landsat_image[5], in_range=(0.05, 0.3), out_range=out_range)
blue = rescale_intensity(landsat_image[6], in_range=(0.025, 0.25), out_range=out_range)
arr_image = np.stack(
    [red, green, blue]
).T
composite = np.rot90(np.flip(arr_image, 1), 1)

# Plot landsat image
plt.imshow(composite)
```

#### 3. Extracting Raster Values at Sample Locations
This is arguably the **most crucial part** for the machine learning classification. It extracts the spectral values from the Landsat imagery precisely at the coordinates of training samples.
  * ```landsat.sample(coords)```: This ```rasterio``` method efficiently samples pixel values from all bands of the ```landsat``` image at each ```(x, y)``` coordinate pair stored in ```coords``` (which came from ```sample_extract``` GeoDataFrame). This means for every training sample point, it gets a vector of spectral values across all Landsat bands.
  * ```sample_extract[["B1", ..., "B9"]] = landsat_extract```: The extracted Landsat spectral values are then assigned to new columns (e.g., "B1", "B2", etc., corresponding to Landsat bands) in ```sample_extract``` GeoDataFrame.
```python
# Extract raster value
landsat_extract = np.stack(
    [x for x in landsat.sample(coords)]
) / 1e4
sample_extract[["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B9"]] = landsat_extract
sample_extract
```

### Load The Palsar data and The Palsar Raster Values Presentation by Extraction
The entire section is **essential for incorporating PALSAR radar imagery into the land cover classification workflow**. It handles the loading, resampling, and crucial extraction of radar backscatter values at the training sample locations, complementing the Landsat optical data.

#### 1. Loading and Resampling PALSAR Imagery
```palsar = rio.open(palsar_dir)```: This line opens the PALSAR image file using rasterio. PALSAR data provides Synthetic Aperture Radar (SAR) information, which is valuable because it can penetrate clouds and provides information about surface roughness and dielectric properties.
```python
# Load palsar image
palsar = rio.open(palsar_dir)
palsar_image = palsar.read(
    out_shape=(palsar.count, landsat_image.shape[1], landsat_image.shape[2]),
    resampling=Resampling.bilinear,
) / 1e3
```

#### 2. Extracting Radar Values at Sample Locations
The **most vital step** for feeding PALSAR data into the machine learning model. It extracts the radar backscatter values from the PALSAR image precisely at the coordinates of training samples.
  * ```sample_extract[["HH", "HV"]] = palsar_extract```: The extracted PALSAR backscatter values are then assigned to new columns (e.g., "HH" for Horizontal-Horizontal polarization, "HV" for Horizontal-Vertical polarization) in  ```sample_extract``` GeoDataFrame. These become additional features for the machine learning classifier.
```python
# Extract raster value
palsar_extract = np.stack([x for x in palsar.sample(coords)]) / 1e3
sample_extract[["HH", "HV"]] = palsar_extract
sample_extract
```

### Split The Sample Extraction
The section is fundamental for preparing data for machine learning model training and evaluation. It performs a crucial step known as data splitting.
This part of the code utilizes the ```train_test_split``` function from ```sklearn.model_selection``` to divide ```sample_extract``` dataset (which contains the extracted satellite imagery features and their corresponding land cover labels) into two distinct subsets: a training set and a testing set.
  * ```train_size=0.7```: This parameter specifies that 70% of data will be used for training the machine learning model. The model will learn patterns and relationships between the satellite imagery features and the land cover classes from this ```train``` subset.
  * ```print(f'Train size: {len(train)}\nTest size: {len(test)}')```: This simply prints the number of samples in both the training and testing sets, providing a quick check on the split.
```python
# Split sample to train and test
seeds = 2
train, test = train_test_split(sample_extract, train_size=0.7, random_state=seeds)
print(f'Train size: {len(train)}\nTest size: {len(test)}')
```

### Land Cover Modelling By Random Forest
The core of the machine learning classification process, where the Random Forest model is defined and trained using prepared training data. It is where the machine learning model actually learns to map the input satellite imagery features to the correct land cover classes.

#### 1. Defining Predictor Variables
These are the columns from ```sample_extract``` (and thus ```train```) DataFrame that contain the actual pixel values extracted from the Landsat and PALSAR imagery.
```python
# Make random forest model
predictors = ["B2", "B3", 'B4', 'B5', 'B6', 'B7', 'HH', 'HV']
```

#### 2. Initializing the Random Forest Classifier
```model = RandomForestClassifier(100)```: This line initializes an instance of the ```RandomForestClassifier``` from ```sklearn.ensemble```. This is an ensemble learning method that operates by constructing a multitude of decision trees during training. For classification tasks, it outputs the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees. ```n_estimators=100```, meaning the Random Forest will build 100 individual decision trees.
```python
model = RandomForestClassifier(100)
```

#### 3. Training the Model
This ```fit``` method the training step of the machine learning model. During this ```fit``` process, the Random Forest algorithm:
  * Randomly samples subsets of the training data (with replacement) to build each of the 100 decision trees.
  * For each tree, it randomly selects a subset of the predictors (features) to consider at each split point.
  * Each tree then learns a set of rules to classify the land cover based on the selected features.
```python
model.fit(
    train[predictors],
    train["value"]
)
```

### Assessment For The Model
Evaluate the performance and reliability of trained land cover classification model. It simulates how the model would perform on new, real-world imagery that it has not encountered before. It allows for an unbiased assessment of the model's generalization capabilities.

#### 1. Making Predictions on the Test Set
```test_apply = model.predict(test[predictors])```: This is the prediction step. Here, the trained ```model``` (```RandomForestClassifier```) is used to predict the land cover class for each sample in the ```test``` dataset.
```python
# Test model
test_apply = model.predict(test[predictors])
```

#### 2. Generating a Confusion Matrix
  * ```cm = confusion_matrix(test['value'], test_apply)```: This calculates the confusion matrix . A confusion matrix is a table that is often used to describe the performance of a classification model on a set of test data for which the true values are known. It shows the number of correct and incorrect predictions made by the classification model, broken down by each class.
```python
# Confusion matrix
cm = confusion_matrix(test['value'], test_apply)
display = ConfusionMatrixDisplay(cm)
display.plot()
```

#### 3. Printing a Classification Report
The classification report is **highly important for a comprehensive quantitative evaluation** because it provides key metrics for each class.
```python
# Report
report = classification_report(test['value'], test_apply)
print(report)
```

### Predicting Land Cover Classification
The trained machine learning model is applied to an entire image to generate the final land cover map, which is then visualized.

#### 1. Predicting Land Cover for the Entire Image
```prediction = model.predict(table_image[predictors])```: This is where trained ```model``` (the ```RandomForestClassifier```) is put to work on the actual, full image data (represented by ```table_image```).
  * ```table_image[predictors]```: This provides the features for each pixel in the entire image. This data would have been prepared in a previous (unshown in this snippet) step, likely by extracting values from the full Landsat and PALSAR images and flattening them into a 2D array suitable for ```model.predict()```.
  * ```prediction```: This variable will store the model's output: a 1D array where each value is the predicted land cover class ID for the corresponding pixel in the input ```table_image```.
```python
# Predict table image
prediction = model.predict(table_image[predictors])
prediction
```

#### 2. Reshaping the Prediction into an Image Format
This step is for **visualization and subsequent geospatial analysis**. The raw 1D array of predictions is not an image; this transformation converts it back into a recognizable image structure that can be plotted and saved as a georeferenced file.
  * ```prediction.reshape(transpose_shape[0], transpose_shape[1])```: The prediction array, being a 1D array of class IDs, needs to be reshaped back into a 2D (height x width) image format. ```transpose_shape``` (presumably derived from the original image dimensions) provides the correct dimensions.
  * ```np.flip(...)``` and ```np.rot90(...)```: These operations apply necessary flips and rotations to correctly orient the reshaped ```prediction``` array, ensuring that the resulting ```prediction_image``` aligns geographically with the original satellite imagery and is not rotated or flipped incorrectly.
```python
# Prediction to image again
prediction_image = np.rot90(np.flip(prediction.reshape(transpose_shape[0], transpose_shape[1]), 1), 1)
```

#### 3. Displaying the Predicted Land Cover Map
This final plotting step is about visual inspection and interpretation of the classification results.
  * ```cmap=cmap```: Applies the **custom colormap** (defined in an earlier section) to the image. This is vital as it colors each land cover class with its predefined, consistent color, making the map immediately interpretable (all water appears blue, all forest appears green).
  * ```interpolation="nearest"```: Specifies the interpolation method for display. "Nearest" is often preferred for discrete classification maps to avoid blurring class boundaries.
```python
# Show to plot
plt.figure(figsize=(10, 10))
plt.imshow(prediction_image, cmap=cmap, interpolation="nearest")
plt.legend(**legend)
```

### Conclusion
This project successfully demonstrates a workflow for automated land cover classification by effectively integrating **remote sensing data (Landsat and PALSAR)** with **machine learning techniques (Random Forest Classifier)**.

How to improve the project: 
  * In traditional random cross-validation (like the ```train_test_split``` used in the current code), data points are randomly assigned to folds, meaning that training and testing samples can be very close to each other geographically. This can lead to an **overestimation of model accuracy**. Therefore, Implementing **Spatial Cross-Validation** is a significant improvement for this land cover classification project, as it directly addresses a critical challenge in geospatial machine learning: **spatial autocorrelation**.
  * **Hyperparameter Tuning**: Optimize the ```RandomForestClassifier```'s hyperparameters (```n_estimators```, ```max_depth```, ```min_samples_leaf```). Techniques like Grid Search or Randomized Search can systematically explore different parameter combinations to find the optimal set for dataset.
  * Explore Other Machine Learning Models: While Random Forest is a strong baseline, consider experimenting with other powerful classifiers: **Support Vector Machines (SVMs)**: Effective for high-dimensional data and can handle complex decision boundaries.



















































































































































