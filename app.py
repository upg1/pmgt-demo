import streamlit as st
from langchain_openai import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Define the OpenAI chain function
def get_openai_chain(api_key: str, template: str):
    openai = OpenAI(api_key=api_key, max_tokens=4000)
    prompt = PromptTemplate(
        input_variables=["input_text"],
        template=template,
    )
    chain = LLMChain(
        llm=openai,
        prompt=prompt,
    )
    return chain

# Function to generate subtasks
def generate_subtasks(goal, api_key):
    template = """
    You are a task management assistant. Given a goal, your task is to break it down into actionable, highly specific subtasks with resources and time intervals e.g. 5 days, 10 weeks, etc.
    Goal: {input_text}
    """
    chain = get_openai_chain(api_key, template)
    result = chain.run({"input_text": goal})
    return result

# Function to generate specific steps for a subtask
def generate_specific_steps(subtask, api_key):
    template = """
    You are a task management assistant. Given a subtask, your task is to break it down into more specific steps over specific shorter time intervals to complete that specific subtask e.g. daily or weekly and specific resources or activities. e.g. compose a schedule for specific activites.
    Subtask: {input_text}
    """
    chain = get_openai_chain(api_key, template)
    result = chain.run({"input_text": subtask})
    return result

# Streamlit app
def main():
    st.title("AI-Powered Task Manager")

    if 'subtasks_list' not in st.session_state:
        st.session_state.subtasks_list = []
    if 'selected_subtask' not in st.session_state:
        st.session_state.selected_subtask = None
    if 'specific_steps' not in st.session_state:
        st.session_state.specific_steps = ""

    api_key = st.text_input("Enter your API key", type="password")

    if api_key:
        goal = st.text_area("Enter your goal")

        if st.button("Generate Subtasks"):
            if goal:
                with st.spinner("Generating subtasks..."):
                    subtasks = generate_subtasks(goal, api_key)
                    subtasks_list = subtasks.split('\n')  # Assuming subtasks are separated by newlines
                    st.session_state.subtasks_list = subtasks_list  # Save subtasks in session state
                    st.session_state.selected_subtask = None  # Reset selected subtask
                    st.session_state.specific_steps = ""  # Reset specific steps
                    st.success("Here are your subtasks:")
            else:
                st.warning("Please enter a goal.")

        # Display subtasks and exploration buttons
        if st.session_state.subtasks_list:
            for subtask in st.session_state.subtasks_list:
                if subtask:
                    if st.button(f"Explore '{subtask}'"):
                        st.session_state.selected_subtask = subtask
                        with st.spinner("Generating specific steps..."):
                            steps = generate_specific_steps(subtask, api_key)
                            st.session_state.specific_steps = steps

        # Display specific steps if a subtask is selected
        if st.session_state.selected_subtask:
            st.text_area(f"Specific Steps for '{st.session_state.selected_subtask}'", value=st.session_state.specific_steps, height=300)

    else:
        st.warning("Please enter your API key.")

if __name__ == "__main__":
    main()