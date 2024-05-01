**Image Compression Project**

This Python project aims to implement an image compression technique using the Discrete Cosine Transform (DCT) algorithm. It offers functionalities for compressing and decompressing images, calculating the Peak Signal-to-Noise Ratio (PSNR), visualizing color components, and plotting PSNR values against different compression levels.

**Usage**

1. **Installation**: Clone the repository to your local machine.

    ```bash
    git clone https://github.com/omarhashem80/2D_DCT
    ```

2. **Requirements**: Ensure you have the necessary dependencies installed. You can install them using pip.

    ```bash
    pip install -r requirements.txt
    ```

3. **Running the Project**: Execute the `run()` method in the `ImageCompressor` class to perform image compression, decompression, PSNR calculation, and plotting.

    ```python
    from ImageCompressor import ImageCompressor

    ImageCompressor.run()
    ```

4. **Output**:
   - Compressed images will be saved in the `Compressed Images` folder.
   - Decompressed images will be saved in the `Decompressed Images` folder.
   - PSNR values for different compression levels will be written to a file named `sizes.txt`.
   - PSNR plot will be saved as `PSNR_plot.png`.
   - Color component visualizations will be saved in the `Image Components` folder.

5. **Customization**: You can modify the input image (`image1.png`) and adjust compression levels in the `run()` method.

**File Structure**

- **ImageCompressor.py**: Contains the implementation of the image compression algorithm.
- **README.md**: Instructions and information about the project.
- **requirements.txt**: List of Python dependencies required for the project.
- **image1.png**: Sample input image for compression.
- **sizes.txt**: File to store compression information and PSNR values.
- **Compressed Images/**: Folder to store compressed images.
- **Decompressed Images/**: Folder to store decompressed images.
- **Image Components/**: Folder to store color component visualizations.
- **PSNR_plot.png**: Plot showing PSNR values against different compression levels.

**Contributing**

Contributions are welcome! If you find any issues or have suggestions for improvements, feel free to open an issue or submit a pull request.

**License**

This project is licensed under the MIT License. See the LICENSE file for details.
