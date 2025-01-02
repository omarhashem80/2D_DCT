# 🌟 **Image Compression Using 2D DCT** 🌟

This Python project demonstrates an image compression technique utilizing the **Discrete Cosine Transform (DCT)** algorithm. It provides functionalities for compressing and decompressing images, calculating the **Peak Signal-to-Noise Ratio (PSNR)**, visualizing color components, and comparing compression performance across different levels.

---

## 🚀 **Features**

1. **Compression & Decompression**  
   - Retains only top-left `m × m` DCT coefficients of each 8x8 block for compression.  
   - Reconstructs images using the inverse DCT (IDCT).  

2. **Performance Metrics**  
   - Calculates **PSNR** values to evaluate image quality after decompression.  
   - Saves detailed compression and decompression sizes for each `m`.  

3. **Visualization**  
   - Visualizes R, G, and B channels of the input image.  
   - Generates a plot of PSNR values for different compression levels.  

4. **Bar Charts**  
   - Compares compressed and decompressed sizes for various `m` values using **Plotly**.

---

## 📁 **File Structure**

- **`ImageCompressor.py`**: Core implementation of the image compression algorithm.  
- **`requirements.txt`**: Python dependencies for the project.  
- **`image1.png`**: Sample input image for demonstration.  
- **`sizes.txt`**: Outputs compression and PSNR data.  
- **`Compressed Images/`**: Stores compressed images.  
- **`Decompressed Images/`**: Stores decompressed images.  
- **`Image Components/`**: Stores color component visualizations.  
- **`PSNRGraph.png`**: Plot showing PSNR values against `m`.  

---

## 🛠️ **Getting Started**

### **1. Clone the Repository**  
```bash
git clone https://github.com/omarhashem80/2D_DCT
cd 2D_DCT
```

### **2. Install Dependencies**  
Use the provided `requirements.txt` to install the necessary libraries.  
```bash
pip install -r requirements.txt
```

### **3. Run the Project**  
Execute the `run()` method from the `ImageCompressor` class to start compression, decompression, and analysis.  
```python
from ImageCompressor import ImageCompressor

ImageCompressor.run('./image1.png', 4)
```

---

## 📤 **Outputs**

- **Compressed Images**: Saved in the `Compressed Images/` folder, with filenames like `compressedImageWithM{m}.png`.  
- **Decompressed Images**: Saved in the `Decompressed Images/` folder.  
- **PSNR Values**: Logged in the `sizes.txt` file for all tested `m` values.  
- **PSNR Plot**: Saved as `PSNRGraph.png`.  
- **Color Components**: Saved in the `Image Components/` folder for R, G, and B channels.  

---


## ⚙️ **Customization**

- Change the **input image** by modifying the `image_path` parameter in the `run()` method.  
- Adjust the **number of retained coefficients (`m`)** to explore different compression levels.  

---

## 🌟 **Acknowledgments**

Thank you for using this project! We hope it helps you understand image compression techniques and inspires further exploration. 🌍✨  

--- 
