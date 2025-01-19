# Load Forecasting Using Machine Learning | 2023
## Overview

This project involves developing a neural network model in TensorFlow for forecasting electrical load with a high accuracy of 89%. Using meteorological data, the model provides accurate demand predictions, which are valuable for power distribution planning and ensuring efficient power usage.

## Table of Contents

[1.Background](#background) 

[2.Installation](#installation)

[3.Usage](#usage)

[4.Model Details](#model-details)

[5.Results](#results)

[6.Acknowledgements](#acknowledgements)

## Background

As an electrical engineering student, this project served as my final year endeavor, focusing on the critical field of load forecasting. Accurate forecasting of electrical load demand is essential for effective power management and distribution, particularly as renewable energy sources become increasingly integrated into the grid.

In this project, I utilized a CSV file containing historical electricity load demand data for three regions, enriched with accompanying meteorological data, such as temperature, rainfall, and humidity. This data serves as the basis for understanding how environmental factors influence electricity consumption patterns.

To analyze and prepare the data, I employed popular Python libraries such as **NumPy** and **Pandas**. These tools enabled me to perform data cleaning, transformation, and exploratory data analysis, allowing me to uncover significant patterns and relationships within the dataset.

For data visualization, I used **Matplotlib** and **Seaborn** to create insightful plots that illustrate trends in electricity demand relative to weather conditions. These visualizations aided in understanding the impact of various factors on load demand and were instrumental in refining the model.

Building upon this analysis, I implemented a neural network using **TensorFlow** and **scikit-learn**. This model was trained to predict electricity load with an accuracy of approximately **89%**. By harnessing the power of machine learning, this project not only demonstrates the application of theoretical knowledge in a practical context but also contributes to enhancing demand forecasting techniques, ultimately aiding in efficient power distribution planning.

## Installation
    
    git clone https://github.com/abhinabgogoi/loadforecast_wth_weather.git
    cd loadforecast_wth_weather

### Install dependencies: Ensure you have Python 3.8+ installed. Then, install the required packages:

    pip install tensorflow pandas numpy matplotlib seaborn scikit-learn`

## Usage

To run the model and see load forecasting results:
![Dataset](https://github.com/user-attachments/assets/7b81a3d1-af7f-48a8-b55e-1d017d76c25a)

1. Prepare the Data: Place the meteorological and load data in the same folder.
2. Run the Notebook: Open the notebook in Jupyter Notebook to explore the data and train the model.
3. Visualize Results: Generated plots, showcase trends, prediction accuracy, and insights.

## Model Details:

  The neural network model developed for this project is designed to predict electrical load demand based on various features derived from the dataset. The model architecture is built using the Keras and accepts an input shape of 15, corresponding to the number of features extracted from the dataset, including historical load data and meteorological variables such as temperature, humidity, and rainfall.

The network consists of three hidden layers, each containing **225 neurons**. This number was chosen based on empirical testing and the complexity of the dataset. The **ReLU (Rectified Linear Unit)** activation function is used for these layers, allowing the model to learn complex patterns by introducing non-linearity. The final output layer consists of a single neuron with a linear activation function, which is appropriate for regression tasks where the goal is to predict a continuous value—in this case, the electrical load demand.

The model is compiled using the **Adam optimizer** with a learning rate of 0.001, and the loss function employed is Mean Squared Error (MSE), which measures the average squared difference between predicted and actual values. Through training on the dataset, this neural network model achieved a prediction accuracy of approximately 89%. This performance demonstrates the effectiveness of using a deep learning approach for load forecasting and highlights the model's capability to learn from historical patterns in the data.

## Results:
![output](https://github.com/user-attachments/assets/37fac92e-ab27-478b-97d8-478e591804f6)

  The developed neural network model provides a reliable forecast of electrical load demand with an accuracy of approximately 89%. This level of accuracy is significant for several reasons. First, it aids in optimizing power distribution by enabling utility companies to allocate resources more efficiently based on predicted demand. By having a clear understanding of expected load patterns, utilities can reduce the risk of overloading their systems and ensure that electricity is delivered where it is needed most.

Additionally, accurate load forecasting contributes to reducing operational costs for utility companies. With precise predictions, utilities can minimize wasteful practices, such as over-generation or under-utilization of resources, leading to more economical operations. Furthermore, the model supports grid stability by aligning supply with demand patterns, which is essential in preventing blackouts and maintaining the reliability of the power supply.

Overall, the model not only demonstrates the technical capabilities of machine learning in load forecasting but also highlights its practical applications in enhancing the efficiency and reliability of power systems.

## Acknowledgements:
  Special thanks to the my project partner, contributors and open-source libraries, including TensorFlow, Pandas, and Matplotlib, for making this project possible.
