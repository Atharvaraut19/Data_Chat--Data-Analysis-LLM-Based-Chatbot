import os
from apikey import apikey
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_openai import OpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from dotenv import load_dotenv, find_dotenv

#OpenAIKey
os.environ['OPENAI_API_KEY'] = apikey
load_dotenv(find_dotenv())


st.title('AI Assistant for Data Science 🤖')


st.write("Hello, 👋 I am your AI Assistant and I am here to help you with your data science projects.")


with st.sidebar:
    st.write('*Data Science Adventure Begins with an CSV File.*')

    st.divider()

    
if 'clicked' not in st.session_state:
    st.session_state.clicked ={1:False}

def clicked(button):
    st.session_state.clicked[button]= True
st.button("Let's get started", on_click = clicked, args=[1])
if st.session_state.clicked[1]:
    user_csv = st.file_uploader("Upload your file here", type="csv")
    if user_csv is not None:
        user_csv.seek(0)
        #Loading of Dataset
        df = pd.read_csv(user_csv, low_memory=False)
        
        #llm model
        llm = OpenAI(temperature=0.1) 


        #Function sidebar
        @st.cache_data
        def steps_eda():
            try:
                steps_eda = llm('What are the steps of EDA')
                return steps_eda
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                return None

        #Pandas agent
        pandas_agent = create_pandas_dataframe_agent(llm, df, verbose = True)

        #Functions main
        @st.cache_data
        def function_agent():
            st.write("**Data Overview**")
            st.write("The first rows of your dataset look like this:")
            st.write(df.head())
            st.write("**Data Cleaning**")
            columns_df = pandas_agent.run("What are the meaning of the columns?")
            st.write(columns_df)
            missing_values = pandas_agent.run("How many missing values does this dataframe have? Start the answer with 'There are'")
            st.write(missing_values)
            duplicates = pandas_agent.run("Are there any duplicate values and if so where?")
            st.write(duplicates)
            st.write("**Data Summarisation**")
            st.write(df.describe())
        
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            if len(numeric_cols) > 1:
                corr = df[numeric_cols].corr()
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
                st.write("**Correlation Heatmap (using numerical columns only)**")
                st.pyplot(fig)
            else:
                st.write("There are not enough numerical columns to calculate a correlation heatmap.")

            try:
                outliers = pandas_agent.run("Identify outliers in the data that may be erroneous or that may have a significant impact on the analysis.")
                st.write(outliers)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

            new_features = pandas_agent.run("What new features would be interesting to create?.")
            st.write(new_features)
            return

        @st.cache_data
        def function_question_variable():
            try:
                summary_statistics = pandas_agent.run(f"Give me a summary of the statistics of {user_question_variable}")
                st.write(summary_statistics)
                normality = pandas_agent.run(f"Check for normality or specific distribution shapes of {user_question_variable}")
                st.write(normality)
                outliers = pandas_agent.run(f"Assess the presence of outliers of {user_question_variable}")
                st.write(outliers)
                trends = pandas_agent.run(f"Analyse trends, seasonality, and cyclic patterns of {user_question_variable}")
                st.write(trends)
                missing_values = pandas_agent.run(f"Determine the extent of missing values of {user_question_variable}")
                st.write(missing_values)
                
                # Generate line Chart
                st.write("**Line Chart**")
                fig, ax = plt.subplots()
                ax.plot(df.index, df[user_question_variable], marker='o', color='b')
                ax.set_xlabel('Index')
                ax.set_ylabel(user_question_variable)
                ax.set_title(f'Line Chart of {user_question_variable}')
                ax.grid(True)
                st.pyplot(fig)
                
                # Generate histogram
                st.write("**Histogram**")
                fig, ax = plt.subplots()
                sns.histplot(data=df, x=user_question_variable, kde=True, bins=20)
                ax.set_xlabel(user_question_variable)
                ax.set_ylabel('Frequency')
                ax.set_title(f'Histogram of {user_question_variable}')
                st.pyplot(fig)
                
                # Generate scatter plot
                st.write("**Scatter Plot**")
                numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                if len(numeric_cols) > 1:
                    fig, ax = plt.subplots()
                    ax = sns.scatterplot(data=df, x=numeric_cols[0], y=user_question_variable)
                    ax.set_xlabel(numeric_cols[0])
                    ax.set_ylabel(user_question_variable)
                    ax.set_title(f'Scatter Plot of {user_question_variable} against {numeric_cols[0]}')
                    ax.grid(True)
                    st.pyplot(fig)
                else:
                    st.write("Insufficient numeric columns to generate Scatter Plot.")
                
                return
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                return None
        
        @st.cache_data
        def function_question_dataframe():
            try:
                dataframe_info = pandas_agent.run(user_question_dataframe)
                st.write(dataframe_info)
                return
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                return None

        #Main

        st.header('Exploratory data analysis')
        st.subheader('General information about the dataset')

        with st.sidebar:
            with st.expander('What are the steps of EDA'):
                st.write(steps_eda())

        function_agent()

        st.subheader('Variable of study')
        user_question_variable = st.text_input('What variable are you interested in')
        if user_question_variable is not None and user_question_variable !="":
            function_question_variable()  #

            st.subheader('Further study')

        if user_question_variable:
            user_question_dataframe = st.text_input( "Is there anything else you would like to know about your dataframe?")
            if user_question_dataframe is not None and user_question_dataframe not in ("","no","No"):
                function_question_dataframe()
            if user_question_dataframe in ("no", "No"):
                st.write("")

        # Ask questions about the dataset
        st.subheader('Ask any questions regarding the dataset')
        user_question = st.text_input('Enter your question about the dataset')
        if user_question is not None and user_question != "":
            try:
                dataset_answer = pandas_agent.run(user_question)
                st.write(dataset_answer)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                
        # Plot Graphs section
        st.header('Plot Graphs')

        # Select graph type
        graph_type = st.selectbox('Select Graph Type', ['Histogram', 'Bar Chart', 'Line Chart', 'Scatter Plot', 'Pie Chart'])

        # Select columns for the graph
        selected_columns = st.multiselect('Select Columns', df.columns)

        categorical_columns = df.select_dtypes(include=['object']).columns

        # Encode categorical columns using one-hot encoding
        df_encoded = pd.get_dummies(df, columns=categorical_columns)

        # Import required libraries for encoding
        from sklearn.preprocessing import LabelEncoder

        # Initialize LabelEncoder
        label_encoder = LabelEncoder()

        # Convert a specific categorical column to numerical using label encoding
        if 'categorical_column' in df.columns:  # Check if the column exists in the DataFrame
            df['categorical_column'] = label_encoder.fit_transform(df['categorical_column'])
                # Plot the graph based on user selection
        if st.button('Plot Graph'):
            if not selected_columns:
                st.warning('Please select at least one column')
            else:
                if graph_type == 'Histogram':
                    for column in selected_columns:
                        fig, ax = plt.subplots()
                        ax.hist(df[column], bins=20)
                        ax.set_xlabel(column)
                        ax.set_ylabel('Frequency')
                        ax.set_title(f'Histogram of {column}')
                        st.pyplot(fig)
                elif graph_type == 'Boxplot':
                    for column in selected_columns:
                        fig, ax = plt.subplots()
                        ax.boxplot(df[column])
                        ax.set_xlabel(column)
                        ax.set_ylabel('Value')
                        ax.set_title(f'Boxplot of {column}')
                        st.pyplot(fig)
                elif graph_type == 'Bar Chart':
                    for column in selected_columns:
                        fig, ax = plt.subplots()
                        ax.bar(df[column].value_counts().index, df[column].value_counts().values)
                        ax.set_xlabel(column)
                        ax.set_ylabel('Frequency')
                        ax.set_title(f'Bar Chart of {column}')
                        st.pyplot(fig)
                elif graph_type == 'Line Chart':
                    for column in selected_columns:
                        fig, ax = plt.subplots()
                        ax.plot(df.index, df[column])
                        ax.set_xlabel('Index')
                        ax.set_ylabel(column)
                        ax.set_title(f'Line Chart of {column}')
                        st.pyplot(fig)
                elif graph_type == 'Scatter Plot':
                    if len(selected_columns) == 2:
                        fig, ax = plt.subplots()
                        ax.scatter(df[selected_columns[0]], df[selected_columns[1]])
                        ax.set_xlabel(selected_columns[0])
                        ax.set_ylabel(selected_columns[1])
                        ax.set_title(f'Scatter Plot between {selected_columns[0]} and {selected_columns[1]}')
                        st.pyplot(fig)
                    else:
                        st.error('Please select exactly two columns for Scatter Plot')
                elif graph_type == 'Pie Chart':
                    for column in selected_columns:
                        fig, ax = plt.subplots()
                        ax.pie(df[column].value_counts(), labels=df[column].value_counts().index, autopct='%1.1f%%')
                        ax.set_title(f'Pie Chart of {column}')
                        st.pyplot(fig)
                    else:
                        st.error('Please select exactly two columns for Stacked Bar Chart')
                else:
                    st.error('Invalid graph type selected')