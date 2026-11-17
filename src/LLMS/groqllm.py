import os
import streamlit as st
from langchain_groq import ChatGroq

class GroqLLM:
    def __init__(self,user_contols_input):
        self.user_controls_input=user_contols_input

    def get_llm_model(self):
        try:
            groq_api_key=self.user_controls_input["GROQ_API_KEY"]
            selected_groq_model=self.user_controls_input["selected_groq_model"]
            if not groq_api_key and not os.environ.get("GROQ_API_KEY"):
                st.error("Please Enter the Groq API KEY")
                return None

            options = {}
            # GPT-OSS does not accept reasoning_format; Qwen reasoning models do.
            if selected_groq_model.startswith('qwen/'):
                options['reasoning_format'] = 'hidden'
            llm=ChatGroq(api_key=groq_api_key or os.environ.get('GROQ_API_KEY'),model=selected_groq_model,**options)

        except Exception:
            raise ValueError('Could not initialize the configured Groq model') from None
        return llm
