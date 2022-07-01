ScleraU-Net2 is developed by the Visual Computing Group (VCG) of Democritus University of Thrace (DUTh) for the participation in the SSBC 2020 competition. It is based on the original U-Net but comprised of a reduced number of layers, leading to a lightweight architecture that is tailored towards sclera segmentation. Key improvements were:

1) Group Normalization (GN) instead of Batch Normalization (BN) after each convolutional layer
2) GELU instead of Relu as activation for all convolutional layers
3) SpatialDropout2D to improve generalization ability during training



In order to run the code, the different datasets used for the competition need to be downloaded separately and placed in the “data” folder following the pattern:

{DATASET_NAME} / {train / validate / test} / {images / masks}
