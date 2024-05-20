Sure, here's the updated README with an embedded video demonstration:

# Data_Chat--Data-Analysis-LLM-Based-Chatbot

## Overview

**Data_Chat** is an interactive data analysis chatbot built using Streamlit and integrated with OpenAI's language model capabilities. This application allows users to upload a CSV file, perform exploratory data analysis (EDA), visualize data through various types of graphs, and get insights powered by a language model. The assistant guides users through the EDA process, answers specific questions about the data, and generates visualizations to help better understand the dataset.

## Features

- **Upload CSV Files**: Users can upload their CSV files to start the analysis.
- **Exploratory Data Analysis (EDA)**: Automatically performs common EDA steps such as data overview, cleaning, summarization, and correlation heatmaps.
- **Custom Queries**: Users can ask specific questions about their dataset and receive detailed answers.
- **Data Visualizations**: Generate various types of graphs including histograms, bar charts, line charts, scatter plots, and pie charts.
- **Interactive Sidebar**: Provides step-by-step guidance and allows users to input variables or ask additional questions.

## Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/your-username/Data_Chat.git
   cd Data_Chat
   ```

2. **Create and activate a virtual environment:**
   ```sh
   python -m venv env
   source env/bin/activate  # On Windows use `env\Scripts\activate`
   ```

3. **Install the required dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

4. **Set up OpenAI API Key:**
   - Create a file named `apikey.py` in the project directory and add your OpenAI API key:
     ```python
     apikey = 'your_openai_api_key_here'
     ```
   - Ensure you have the `.env` file with the necessary environment variables:
     ```
     OPENAI_API_KEY=your_openai_api_key_here
     ```

## Usage

1. **Run the Streamlit application:**
   ```sh
   streamlit run app.py
   ```

2. **Upload your CSV file:**
   - Click on the "Let's get started" button.
   - Use the file uploader in the sidebar to upload your CSV file.

3. **Perform EDA:**
   - The application will automatically display an overview of the data.
   - Check for missing values, duplicates, and basic statistics.
   - View the correlation heatmap if there are enough numerical columns.

4. **Ask Questions:**
   - Enter specific questions about the dataset in the input box provided.
   - Receive insights and visualizations based on the language model's analysis.

5. **Visualize Data:**
   - Choose the type of graph you want to generate from the dropdown menu.
   - Select the columns you wish to visualize.
   - Click on the "Plot Graph" button to generate and display the graph.

## Project Structure

- `app.py`: Main Streamlit application file.
- `apikey.py`: File containing the OpenAI API key.
- `requirements.txt`: List of required Python packages.
- `.env`: Environment variables for the application.

## Requirements

- **Python 3.7 or higher**
- **Streamlit**
- **Pandas**
- **Matplotlib**
- **Seaborn**
- **OpenAI Python SDK**
- **LangChain**

To install all required packages, use the command:
```sh
pip install -r requirements.txt
```



## Video Demonstration

For a video demonstration of the Data_Chat application, watch the video named as Application Video.

---

Enjoy your data science adventure with Data_Chat! If you have any questions or need further assistance, feel free to reach out. Happy analyzing!